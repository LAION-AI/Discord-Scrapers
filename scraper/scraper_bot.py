import io
import os
import time
import json
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List, Dict, Any
from huggingface_hub import HfFileSystem, preupload_lfs_files, create_commit, CommitOperationCopy, CommitOperationDelete, CommitOperationAdd

import requests
from PIL import Image as PILImage
from datasets import Dataset, Image
from dataclasses import fields

# Load environment variables from .env file if they exist
# Mainly used for local development, github actions will set these variables on its own.
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if line.strip():
                key, value = line.strip().split('=', 1)
                if value:
                    print(f"Setting {key} from .env file")
                    os.environ[key] = value
else:
    print(".env file not found, skipping.")


@dataclass
class ScraperBotConfig:
    """Bot configuration that changes how the bot behaves"""
    base_url: str
    channel_id: str
    limit: int
    max_chunk_size: int
    hf_dataset_name: str

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
            return cls(**config_dict)


@dataclass(frozen=True)
class HFDatasetScheme:
    caption: str
    image: Image(decode=True)
    link: str
    message_id: str
    timestamp: str


def prepare_dataset(messages: List[HFDatasetScheme]) -> pd.DataFrame:
    return pd.DataFrame({
            "caption": [msg.caption for msg in messages],
            "image": [None for msg in messages],  # Initialize to None, will be filled in later
            "link": [msg.link for msg in messages],  # will maintain just because we use it to filter
            "message_id": [msg.message_id for msg in messages],
            "timestamp": [msg.timestamp for msg in messages]
    })


# Download images
def download_and_convert_images_for_dataframe(df: pd.DataFrame) -> dict:
    # Convert the DataFrame to a dictionary
    df_dict = df.to_dict('list')

    # Initialize dataset_dict with existing data
    dataset_dict = {key: df_dict[key][:] for key in df_dict.keys()}

    # Identify rows with NA in the 'image' column
    rows_to_download = [idx for idx, image in enumerate(df_dict['image']) if image is None]
    count_to_download = len(rows_to_download)
    print(f"Downloading {count_to_download} new images...")

    # Loop to download images
    for idx in tqdm(rows_to_download, desc="Downloading images", unit=" images"):
        image_url = df_dict['link'][idx]
        try:
            image = PILImage.open(requests.get(image_url, stream=True).raw).convert("RGB")
            dataset_dict['image'][idx] = image  # Replace the None value
        except Exception as e:
            print(f"Error downloading image at {image_url}: {e}")

    return dataset_dict


def merge_datasets(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # Gather existing URLs from old_df using indices if it exists
    existing_image_urls = old_df['link'].tolist() if old_df is not None else []

    print(f"Current rows: {len(existing_image_urls)}")

    # Filter new_df to only include rows that aren't in old_df
    filtered_new_df = new_df.loc[~new_df['link'].isin(existing_image_urls)]

    print(f"Rows to add: {filtered_new_df.shape[0]}")

    # Concatenate the old and filtered new dataframes
    merged_df = pd.concat([old_df, filtered_new_df]) if old_df is not None else filtered_new_df

    return merged_df


def get_latest_message_id(df: pd.DataFrame) -> str:
    try:
        max_message = df['message_id'].astype(np.int64).max()
        if math.isnan(max_message):
            return None
        return str(max_message)
    except Exception as e:
        print(e)
        return None


def get_bot_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bot {os.environ['DISCORD_TOKEN']}"
    }


def get_user_headers() -> Dict[str, str]:
    return {
        'authorization': os.environ['DISCORD_TOKEN']
    }


def get_image(link: str) -> bytes:
    image = PILImage.open(requests.get(link, stream=True).raw).convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return {'bytes': img_byte_arr.getvalue(), 'path': None}


fs = HfFileSystem(token=os.environ['HF_TOKEN'])

schema = [f.name for f in fields(HFDatasetScheme)]


class ScraperBot:
    def __init__(self, config: ScraperBotConfig, condition_fn: Callable, parse_fn: Callable) -> None:
        """A bot that scrapes messages from a Discord channel and uploads them to the Hugging Face Hub.

        Parameters
        ----------
        config : ScraperBotConfig
            Configuration for the bot.
        condition_fn : Callable
            A function that receives a message with type Dict[str, Any] and returns a boolean.
        parse_fn : Callable
            A function that receives a message with type Dict[str, Any] and returns a parsed
            message with type List[HFDatasetScheme].
        """
        self.config = config
        self.base_url = config.base_url
        self.channel_id = config.channel_id
        self.limit = config.limit
        self.hf_dataset_name = config.hf_dataset_name
        self.parse_fn = parse_fn
        self.condition_fn = condition_fn

    @property
    def url(self) -> str:
        return f'{self.base_url}/channels/{self.channel_id}/messages?limit={self.limit}'

    @property
    def headers(self) -> Dict[str, str]:
        if os.environ['IS_CI'] in 'true':
            return get_bot_headers()
        else:
            return get_user_headers()

    @property
    def fs_path(self) -> str:
        return f"datasets/{self.hf_dataset_name}/data"

    @property
    def repo_path(self) -> str:
        return "data"

    def _get_chunk_names(self) -> None:
        chunks = fs.glob(f"{self.fs_path}/**.parquet")
        return [chunk.replace(f"{self.fs_path}/", '') for chunk in chunks]

    def _update_chunk(self, df: pd.DataFrame, chunk_num: int) -> None:
        chunks = self._get_chunk_names()
        number_of_chunks = len(chunks)

        # Save the current chunk
        with fs.open(f"{self.fs_path}/train-{chunk_num:04d}-of-{number_of_chunks + 1:04d}.parquet", "wb") as f:
            df.to_parquet(f)

    def _new_chunk(self, df: pd.DataFrame) -> None:
        # Rename all chunks to be of one number higher
        chunks = self._get_chunk_names()
        # re https://github.com/huggingface/huggingface_hub/issues/1733#issuecomment-1761942073,
        # a commit should not have more than 100 operations (so not more than 50 files should be renamed at once).
        # The issue is being timed out. testing shows that it should be fine for many rename operations.
        # hf_hub only has CommitOperationCopy and CommitOperationDelete,
        # but we can combine them into a CommitOperationRename
        new_chunk_count = len(chunks) + 1
        operations = []
        for chunk in chunks:
            key = int(chunk.split("-")[1])
            from_name = f"{self.repo_path}/{chunk}"
            to_name = f"{self.repo_path}/train-{key:04d}-of-{new_chunk_count:04d}.parquet"
            operations.append(CommitOperationCopy(from_name, to_name))
            operations.append(CommitOperationDelete(from_name))

        addition = CommitOperationAdd(
            path_in_repo=f"{self.repo_path}/train-{len(chunks):04d}-of-{new_chunk_count:04d}.parquet",
            path_or_fileobj=df.to_parquet()
        )
        preupload_lfs_files(repo_id=self.hf_dataset_name, repo_type='dataset', token=os.environ['HF_TOKEN'],  additions=[addition])
        operations.append(addition)

        create_commit(
            repo_id=self.hf_dataset_name,
            repo_type='dataset',
            commit_message="Rename chunks",
            token=os.environ['HF_TOKEN'],
            operations=operations,
        )
        # split into multiple commits. We can't split the copy and delete operations into separate commits

    def _load_dataset(self) -> (Dataset, int):
        chunks = self._get_chunk_names()
        if len(chunks) == 0:
            return (None, 0)
        # sort in descending order. The naming scheme is  train-<x>-of-<y>-<hash>.parquet
        chunks.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
        df = pd.read_parquet(fs.open(f"{self.fs_path}/{chunks[0]}", "rb"))
        chunk_num = int(chunks[0].split("-")[1])

        # This is a temporary fix for the transition period between the old and new saving schemes
        # Search all chunks for the max message id
        # TODO: Remove this once we are sure that all datasets have a new chunk - that guarantees that the latest message id is in the last chunk
        latest_message_id = '0'
        for chunk in chunks:
            _df = pd.read_parquet(fs.open(f"{self.fs_path}/{chunk}", "rb"))
            _latest_message_id = get_latest_message_id(df)
            if _latest_message_id > latest_message_id:
                latest_message_id = _latest_message_id
                df = _df
                chunk_num = int(chunk.split("-")[1])

        return (df, chunk_num)

    def filter_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Iterate over the whole dataset and remove all messages that are already in the dataset
        chunks = self._get_chunk_names()
        if len(chunks) == 0:
            return messages

        for chunk in tqdm(chunks, desc="Filtering messages", unit=" chunks"):
            df = pd.read_parquet(fs.open(f"{self.fs_path}/{chunk}", "rb"))
            existing_message_ids = df['message_id'].tolist()
            messages = [msg for msg in messages if msg['message_id'] not in existing_message_ids]

        return messages

    def _get_messages(self, after_message_id: str) -> List[Dict[str, Any]]:
        all_messages = []
        before_message_id = None

        progress = tqdm(desc="Fetched messages", unit=" messages")

        while True:
            url = self.url
            if before_message_id:
                url += f'&before={before_message_id}'
            if after_message_id:
                url += f'&after={after_message_id}'

            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                messages = response.json()

                if not messages:
                    break

                parsed_messages = [self.parse_fn(msg) for msg in messages if self.condition_fn(msg)]
                parsed_messages = [msg for msg_list in parsed_messages for msg in msg_list]

                all_messages.extend(parsed_messages)

                # Update tqdm progress bar
                progress.update(len(parsed_messages))

                if messages[-1]['id'] == before_message_id:
                    break

                before_message_id = messages[-1]['id']
            elif response.status_code == 429:
                print("Rate limited. Sleeping for 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to fetch messages. Response: {response.json()}")
                break

        # Close the tqdm progress bar
        progress.close()

        unique_objects = set()
        unique_list = []
        for obj in all_messages:
            if obj not in unique_objects:
                unique_objects.add(obj)
                unique_list.append(obj)
        unique_list.sort(key=lambda x: x.message_id)
        return unique_list

    def scrape(self, fetch_all: bool = False, push_to_hub: bool = True) -> Dataset:
        (chunk, chunk_num) = self._load_dataset()
        if chunk is None:
            print("No existing dataset found.")
            chunk = pd.DataFrame(columns=schema)
            after_message_id = None
        else:
            after_message_id = get_latest_message_id(chunk) if not fetch_all else None
        messages = self._get_messages(after_message_id=after_message_id)
        messages = self.filter_messages(messages) if fetch_all else messages
        messages = prepare_dataset(messages)

        if not len(messages):
            print("No new messages found.")
            return

        for index, row in tqdm(messages.iterrows()):
            if len(chunk) >= self.config.max_chunk_size:
                self._update_chunk(chunk, chunk_num)
                time.sleep(5)  # Sleep for 5 seconds to avoid race conditions
                # Save the current chunk
                chunk = pd.DataFrame(columns=schema)
                chunk_num += 1
                self._new_chunk(chunk)

            try:
                row['image'] = get_image(row['link'])
                chunk = pd.concat([chunk, pd.DataFrame([row])], ignore_index=True)
            except Exception as e:
                print(f"Error downloading image at {row['link']}: {e}")

        time.sleep(5)  # Sleep for 5 seconds to avoid race conditions
        self._update_chunk(chunk, chunk_num)
