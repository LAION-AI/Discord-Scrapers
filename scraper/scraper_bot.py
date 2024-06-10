import os
import time
import json
import math
import pandas as pd
import numpy as np
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any
from huggingface_hub import (
    HfFileSystem,
    upload_file,
    hf_hub_download,
    create_commit,
    create_repo,
    CommitOperationCopy,
    CommitOperationDelete,
    DatasetCard,
)
from datasets import (
    Dataset,
    Image,
    load_dataset,
    disable_caching,
    info,
    splits,
    Image,
    Audio,
    Value,
    Features,
    utils,
)
import requests
from PIL import Image as PILImage
from dataclasses import fields

disable_caching()


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


# Load environment variables from .env file if they exist
# Mainly used for local development, github actions will set these variables on its own.
@run_once
def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.strip():
                    key, value = line.strip().split("=", 1)
                    if value:
                        print(f"Setting {key} from .env file")
                        os.environ[key] = value
    else:
        print(".env file not found, skipping.")


load_env()


@dataclass
class ScraperBotConfig:
    """Bot configuration that changes how the bot behaves"""

    base_url: str
    channel_id: str
    limit: int
    max_chunk_size: int
    embed_data: bool
    data_key: str
    hf_dataset_name: str

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            config_dict = json.load(f)
            return cls(**config_dict)


# Dummy class to use as a type hint
@dataclass(frozen=True)
class HFDatasetScheme:
    pass


class AppendMode(Enum):
    LATEST = "latest"
    NEW = "new"


def merge_datasets(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # Gather existing URLs from old_df using indices if it exists
    existing_data_urls = old_df["link"].tolist() if old_df is not None else []

    print(f"Current rows: {len(existing_data_urls)}")

    # Filter new_df to only include rows that aren't in old_df
    filtered_new_df = new_df.loc[~new_df["link"].isin(existing_data_urls)]

    print(f"Rows to add: {filtered_new_df.shape[0]}")

    # Concatenate the old and filtered new dataframes
    merged_df = (
        pd.concat([old_df, filtered_new_df]) if old_df is not None else filtered_new_df
    )

    return merged_df


def get_latest_message_id(df: pd.DataFrame) -> str:
    try:
        max_message = df["message_id"].astype(np.int64).max()
        if math.isnan(max_message):
            return None
        return str(max_message)
    except Exception as e:
        print(e)
        return None


def get_bot_headers() -> Dict[str, str]:
    return {"Authorization": f"Bot {os.environ['DISCORD_TOKEN']}"}


def get_user_headers() -> Dict[str, str]:
    return {"authorization": os.environ["DISCORD_TOKEN"]}


class ScraperBot:
    def __init__(
        self,
        config: ScraperBotConfig,
        HFDatasetScheme,
        prepare_dataset: Callable,
        condition_fn: Callable,
        parse_fn: Callable,
        download_fn: Callable,
        hash_fn: Optional[Callable] = None,
        readme_template: str = "dataset_readme_template.md",
    ) -> None:
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
        self.HFDatasetScheme = HFDatasetScheme
        self.prepare_dataset = prepare_dataset
        self.config = config
        self.base_url = config.base_url
        self.channel_id = config.channel_id
        self.limit = config.limit
        self.embed_data = config.embed_data
        self.data_key = config.data_key
        self.hf_dataset_name = config.hf_dataset_name
        self.parse_fn = parse_fn
        self.condition_fn = condition_fn
        self.download_fn = download_fn
        self.hash_fn = hash_fn
        self.readme_template = readme_template

    @property
    def schema(self):
        schema = [f.name for f in fields(self.HFDatasetScheme)]
        if not self.embed_data:
            schema.remove(self.data_key)
        return schema

    @property
    def url(self) -> str:
        return f"{self.base_url}/channels/{self.channel_id}/messages?limit={self.limit}"

    @property
    def headers(self) -> Dict[str, str]:
        if os.environ["IS_CI"] in "true":
            return get_bot_headers()
        else:
            return get_user_headers()

    @property
    def fs_path(self) -> str:
        path = f"datasets/{self.hf_dataset_name}/{self.repo_path}"
        return path

    @property
    def repo_path(self) -> str:
        return "data"

    def _create_repo(self) -> None:
        print(f"Repo did not exist, creating new repo {self.hf_dataset_name}")
        create_repo(
            self.hf_dataset_name,
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
            exist_ok=True,
        )
        self._create_readme()

    def _create_readme(self) -> None:
        upload_file(
            path_or_fileobj=os.path.join(
                os.path.dirname(__file__), self.readme_template
            ),
            path_in_repo="README.md",
            repo_id=self.hf_dataset_name,
            token=os.environ["HF_TOKEN"],
            repo_type="dataset",
        )

    def _update_readme(self, ds) -> None:
        dataset_card_path = hf_hub_download(
            self.hf_dataset_name, "README.md", repo_type="dataset"
        )
        dataset_card = DatasetCard.load(Path(dataset_card_path))
        dataset_card = DatasetCard.load(self.hf_dataset_name)
        dataset_card_data = dataset_card.data
        info.DatasetInfosDict({"default": ds.info}).to_dataset_card_data(
            dataset_card_data
        )

        info_to_dump: info.DatasetInfo = ds.info.copy()
        info_to_dump.config_name = "default"
        info_to_dump.splits = splits.SplitDict()

        total_dataset_nbytes = ds._estimate_nbytes()
        info_to_dump.splits.add(splits.SplitInfo(
            str('train'), num_bytes=total_dataset_nbytes, num_examples=len(ds)
        ))
        info_to_dump.download_checksums = None
        info_to_dump.download_size = total_dataset_nbytes
        info_to_dump.dataset_size = total_dataset_nbytes
        info_to_dump.size_in_bytes = total_dataset_nbytes

        info.DatasetInfosDict({"default": info_to_dump}).to_dataset_card_data(
            dataset_card_data
        )

        dataset_card.data = dataset_card_data

        upload_file(
            path_or_fileobj=str(dataset_card).encode(),
            path_in_repo="README.md",
            repo_id=self.hf_dataset_name,
            token=os.environ["HF_TOKEN"],
            repo_type="dataset",
            commit_message="Update README.md with new dataset info",
        )

    def _get_chunk_names(self) -> None:
        fs = HfFileSystem(token=os.environ["HF_TOKEN"], skip_instance_cache=True)
        try:
            chunks = fs.glob(f"{self.fs_path}/**.parquet")
            return [chunk.replace(f"{self.fs_path}/", "") for chunk in chunks]
        except FileNotFoundError:
            return []

    def _get_detailed_chunk_names(self) -> None:
        fs = HfFileSystem(token=os.environ["HF_TOKEN"], skip_instance_cache=True)
        detailed_chunks = fs.ls(f"{self.fs_path}/", detail=True)
        return detailed_chunks

    def _append_chunk(
        self, df: pd.DataFrame, mode: AppendMode = AppendMode.LATEST
    ) -> None:
        fs = HfFileSystem(token=os.environ["HF_TOKEN"], skip_instance_cache=True)
        chunks = self._get_chunk_names()
        print(f"Appending to {len(chunks)} existing chunks with mode {mode}")

        if mode == AppendMode.NEW or not chunks:
            total = len(chunks) + 1
            key = len(chunks)
            selected_chunk = f"train-{key:05d}-of-{total:05d}"
            print(f"Creating new chunk: {selected_chunk}")
        else:
            most_recent_chunk = max(chunks, key=lambda x: int(x.split("-")[1]))
            key = int(most_recent_chunk.split("-")[1])
            total_parts = most_recent_chunk.split("-")[3]
            total = (
                int(total_parts.split(".")[0])
                if "." in total_parts
                else int(total_parts)
            )
            selected_chunk = f"train-{key:05d}-of-{total:05d}"
            print(f"Updating existing chunk: {selected_chunk}")

        # Prepare and upload the DataFrame
        needs_upload = True
        while needs_upload:
            df = df.reset_index(drop=True)  # drop index before converting to dataset
            ds = Dataset.from_pandas(df)

            if "image" in ds.column_names:
                ds = ds.cast_column("image", Image(decode=True))
            if "audio" in ds.column_names:
                ds = ds.cast_column("audio", Audio(decode=True, mono=False))

            file_name = f"{self.fs_path}/{selected_chunk}-{ds._fingerprint}.parquet"
            print(f"Saving chunk {file_name} with {df.shape[0]} rows")
            try:
                with fs.open(file_name, "wb") as f:
                    ds.to_parquet(f)
                needs_upload = False
            except Exception as e:
                print(f"Upload failed {e}, retrying...")
                time.sleep(5)

            # Update the readme on success
            # FIXME: calculate new ds size from parquet files
            # Currently this only find the size of this single chunk
            # self._update_readme(ds.info)

    def _rename_chunks(self):
        # Rename all chunks to be of one number higher
        chunks = self._get_detailed_chunk_names()
        # re https://github.com/huggingface/huggingface_hub/issues/1733#issuecomment-1761942073,
        # a commit should not have more than 100 operations (so not more than 50 files should be renamed at once).
        # The issue is being timed out. testing shows that it should be fine for many rename operations.
        # hf_hub only has CommitOperationCopy and CommitOperationDelete,
        # but we can combine them into a CommitOperationRename
        operations = []
        fingerprints = []

        # First pass to identify and delete smaller duplicates
        processed_base_names = set()
        for chunk in chunks:
            name = chunk.get("name").replace(f"{self.fs_path}/", "")
            key = int(name.split("-")[1])
            base_name = f"train-{key:05d}"

            if base_name in processed_base_names:
                continue

            duplicate_chunks = [
                other_chunk
                for other_chunk in chunks
                if other_chunk.get("name")
                .replace(f"{self.fs_path}/", "")
                .startswith(base_name)
            ]

            if len(duplicate_chunks) > 1:
                print(
                    f"Found {len(duplicate_chunks)} duplicate chunks with base name {base_name}"
                )
                smallest_chunk = min(duplicate_chunks, key=lambda x: x.get("size"))
                print(
                    f"Deleting smaller chunk {smallest_chunk.get('name').replace(f'{self.fs_path}/', '')}"
                )
                operations.append(
                    CommitOperationDelete(
                        f"{self.repo_path}/{smallest_chunk.get('name').replace(f'{self.fs_path}/', '')}"
                    )
                )

                # Remove the smallest chunk from the chunks list and mark the base_name as processed
                chunks.remove(smallest_chunk)
                processed_base_names.add(base_name)

        new_chunk_count = len(chunks)

        # Second pass to rename chunks
        for chunk in chunks:
            name = chunk.get("name").replace(f"{self.fs_path}/", "")
            key = int(name.split("-")[1])

            split_name = name.split(".")[0].split("-")
            if len(split_name) == 5:
                fingerprint = split_name[-1]
            else:
                fingerprint = None

            from_name = f"{self.repo_path}/{name}"

            if fingerprint:
                to_name = f"{self.repo_path}/train-{key:05d}-of-{new_chunk_count:05d}-{fingerprint}.parquet"
            else:
                to_name = (
                    f"{self.repo_path}/train-{key:05d}-of-{new_chunk_count:05d}.parquet"
                )

            if from_name == to_name:
                print(
                    f"Skipping chunk {from_name} because it is already named correctly"
                )
                continue

            print(f"Renaming chunk {from_name} to {to_name}")

            if fingerprint and fingerprint in fingerprints:
                raise ValueError(
                    f"Duplicate fingerprint {fingerprint} found, something is wrong"
                )

            if fingerprint:
                fingerprints.append(fingerprint)

            operations.append(CommitOperationCopy(from_name, to_name))
            operations.append(CommitOperationDelete(from_name))

        create_commit(
            repo_id=self.hf_dataset_name,
            repo_type="dataset",
            commit_message="Rename chunks",
            token=os.environ["HF_TOKEN"],
            operations=operations,
        )

    def _load_dataset(self, schema: dict) -> (Dataset, int):
        chunks = self._get_chunk_names()
        if len(chunks) == 0:
            # Initialize the dataset if empty or non-existent
            self._create_repo()
            return (None, 0)

        # sort in descending order.
        # The naming scheme is  train-<x>-of-<y>-<hash>.parquet
        chunks.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
        chunk_num = len(chunks)  # y in the naming scheme
        print(f"Found {len(chunks)} chunks: {chunks}")

        if self.data_key in schema:
            schema.remove(self.data_key)

        print(
            f"Loading and converting to Hugging Face Dataset with columns {schema}..."
        )
        ds = load_dataset(
            self.hf_dataset_name,
            columns=schema,
            split="train",
            streaming=True,
            verification_mode="no_checks",
        )

        df = pd.DataFrame(ds)

        return (df, chunk_num)

    def filter_messages(
        self, dataset: pd.DataFrame, messages: List[HFDatasetScheme]
    ) -> List[HFDatasetScheme]:

        # Iterate over the whole dataset and remove all messages that are already in the dataset
        chunks = self._get_chunk_names()
        if len(chunks) == 0:
            return messages

        existing_message_ids = dataset["message_id"].tolist()
        messages = [
            msg for msg in messages if msg.message_id not in existing_message_ids
        ]

        return messages

    def _get_current_chunk(self) -> pd.DataFrame:
        fs = HfFileSystem(token=os.environ["HF_TOKEN"], skip_instance_cache=True)

        # Iterate over the whole dataset and remove all messages that are already in the dataset
        chunks = self._get_chunk_names()

        current_chunk = pd.DataFrame(columns=self.schema)
        if len(chunks) == 0:
            return current_chunk

        # Find and load current chunk
        chunks.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
        latest_chunk_name = chunks.pop(0)
        latest_chunk = pd.read_parquet(
            fs.open(f"{self.fs_path}/{latest_chunk_name}", "rb")
        )
        print(f"Latest chunk {latest_chunk_name} has {latest_chunk.shape[0]} rows")

        return latest_chunk

    def _get_messages(self, after_message_id: str) -> List[HFDatasetScheme]:
        all_messages = []
        before_message_id = None

        progress = tqdm(desc="Fetching messages", unit=" messages")
        total_messages = 0

        while True:
            url = self.url
            if before_message_id:
                url += f"&before={before_message_id}"
            if after_message_id:
                url += f"&after={after_message_id}"

            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                messages = response.json()
                total_messages += len(messages)

                if not messages:
                    break

                parsed_messages = [
                    self.parse_fn(msg) for msg in messages if self.condition_fn(msg)
                ]
                parsed_messages = [
                    msg for msg_list in parsed_messages for msg in msg_list
                ]

                all_messages.extend(parsed_messages)

                # Update tqdm progress bar
                progress.update(len(messages))

                if messages[-1]["id"] == before_message_id:
                    break

                before_message_id = messages[-1]["id"]
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
        print(
            f"Found {len(unique_list)} valid samples out of {total_messages} messages."
        )
        if unique_list:
            print(f"Oldest: {unique_list[0].timestamp}")
            print(f"Newest: {unique_list[-1].timestamp}")

        return unique_list

    def scrape(self, fetch_all: bool = False) -> None:
        schema = [f.name for f in fields(self.HFDatasetScheme)]

        # Drop data if we're not embedding them
        if not self.embed_data:
            schema.remove(self.data_key)

        print(
            f"Beginning scrape for {self.hf_dataset_name} with schema {schema} and fetch_all={fetch_all}"
        )

        # Load the current dataset without data initially, to figure out what we're working with
        current_dataset, chunk_count = self._load_dataset(schema=schema)
        after_message_id = (
            get_latest_message_id(current_dataset) if not fetch_all else None
        )

        print(
            f"Current dataset has {current_dataset.shape[0] if current_dataset is not None else 0} rows and {chunk_count} chunks."
        )
        print(f"Last message ID: {after_message_id}.")

        # set to true to update all messages from history
        # note that this will add back any previously manually removed entries
        # fetch_all = True 
        messages = self._get_messages(
            after_message_id=after_message_id if not fetch_all else None
        )

        # Filter messages
        filtered_messages = self.filter_messages(current_dataset, messages)

        # Early return if no new messages
        if not len(filtered_messages):
            print("No new messages found.")
            return

        new_message_dataset = self.prepare_dataset(filtered_messages)

        print(
            f"New data has {len(new_message_dataset['link'])} rows and {len(new_message_dataset['link']) // self.config.max_chunk_size + 1} chunks."
        )
        print(
            f"New + Current dataset will have {len(new_message_dataset) + len(current_dataset) if current_dataset is not None else len(new_message_dataset)} rows."
        )
        print(f"Schema: {self.schema}")

        total_rows = len(new_message_dataset)
        current_chunk = self._get_current_chunk()
        schema = current_chunk.columns

        print(f"Initial current_chunk size: {len(current_chunk)}")

        if current_chunk.shape[0] >= self.config.max_chunk_size:
            # If the current chunk is full, create a new one
            print(f"Current chunk is full, starting new chunk...")
            current_chunk = pd.DataFrame(columns=self.schema)
            self._append_chunk(current_chunk, mode=AppendMode.NEW)
            self._rename_chunks()

            print(f"Current current_chunk size is now: {len(current_chunk)}")

        for index, row in tqdm(
            new_message_dataset.iterrows(),
            desc="Uploading to hf hub",
            unit=" rows",
            total=total_rows,
        ):
            # Add the data and append the row to the row_buffer
            try:
                if self.embed_data:
                    row[self.data_key] = self.download_fn(row["link"])
                    if "image_hash" in row and "bytes" in row[self.data_key]:
                        if self.hash_fn:
                            row["image_hash"] = self.hash_fn(row[self.data_key]['bytes'])
                            print(f"Image hash: {row['image_hash']}")
                current_chunk = pd.concat(
                    [current_chunk, pd.DataFrame([row])], ignore_index=True
                )
            except Exception as e:
                print(f"Error downloading data at {row['link']}: {e}")
                continue

            if current_chunk.shape[0] >= self.config.max_chunk_size:
                # If the current chunk is full, create a new one
                print("Current chunk is full, saving and starting new chunk...")
                print("Appending to latest chunk...")
                self._append_chunk(current_chunk, mode=AppendMode.LATEST)
                self._rename_chunks()
                print("Starting new chunk...")
                current_chunk = pd.DataFrame(columns=schema)
                current_chunk.reset_index(drop=True, inplace=True)
                self._append_chunk(current_chunk, mode=AppendMode.NEW)
                self._rename_chunks()

        # Loop finished, check if there is any data left in the current_chunk
        if len(current_chunk) > 0:
            print(f"Current chunk has {len(current_chunk)} rows left, saving...")
            self._append_chunk(current_chunk, mode=AppendMode.LATEST)
            self._rename_chunks()

        print("Done!")
