import os
import time
import json
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional, Any

import requests
from PIL import Image as PILImage
from datasets import load_dataset, Dataset, concatenate_datasets, Image, Value, Features
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
        
    print(f"Merged DataFrame rows before return: {merged_df.shape[0]}")
    return merged_df


def get_latest_message_id(df: pd.DataFrame) -> str:
    try:
        return df["message_id"].max()
    except:
        pass


def get_bot_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bot {os.environ['DISCORD_TOKEN']}"
    }


def get_user_headers() -> Dict[str, str]:
    return {
        'authorization': os.environ['DISCORD_TOKEN']
    }

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


    def _get_messages(self, after_message_id: str) -> List[Dict[str, Any]]:
        all_messages = []
        before_message_id = None
        
        progress = tqdm(desc="Fetching messages", unit=" messages")

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

        return unique_list


    def scrape(self, fetch_all: bool=False, push_to_hub: bool=True) -> Dataset:
        schema = [f.name for f in fields(HFDatasetScheme)]

        print(f"Beginning scrape for {self.hf_dataset_name} with schema {schema}")

        try:
            current_dataset = load_dataset(self.hf_dataset_name)['train'].to_pandas()[schema]
            after_message_id = get_latest_message_id(current_dataset) if not fetch_all else None
            print(f"Current dataset has {current_dataset.shape[0]} rows. Last message ID: {after_message_id}.")
        except Exception as e:
            current_dataset = None
            after_message_id = None
            print(f"No existing dataset found. {e}")
    
        messages = self._get_messages(after_message_id=after_message_id)
        print(f"Fetched {len(messages)} messages.")

        new_dataset = prepare_dataset(messages)

        # Merge the new datafrane with the existing dataframe
        if current_dataset is not None:
            df = merge_datasets(current_dataset, new_dataset)
            if df.shape[0] == current_dataset.shape[0]:
                print("No new rows. Exiting...")
                return
        else:
            df = new_dataset

        # Transform df_dict so that each key maps to a list of values
        dataset_dict = download_and_convert_images_for_dataframe(df)

        # Get the new dataset size
        print(f"New dataset has {len(dataset_dict['link'])} rows and schema: {dataset_dict.keys()}")

        # Convert to Hugging Face Dataset
        print(f"Converting to Hugging Face Dataset...")    
        ds = Dataset.from_dict(dataset_dict)

        # Push to the Hugging Face Hub
        print(f"Pushing dataset to the Hugging Face Hub...")
        print(ds)

        if push_to_hub:
            ds.push_to_hub(self.hf_dataset_name, token=os.environ['HF_TOKEN'])

        return ds