import os
import json
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional, Any

import requests
from PIL import Image
from datasets import load_dataset, Dataset, concatenate_datasets

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
    link: str
    message_id: str
    timestamp: str
    

def prepare_dataset(messages: List[HFDatasetScheme]) -> Dataset:
    return Dataset.from_dict(
        {
            "caption": [msg.caption for msg in messages],
            "image": [Image.open(requests.get(msg.link, stream=True).raw).convert("RGB") for msg in messages],
            "link": [msg.link for msg in messages], # will maintain just because we use it to filter
            "message_id": [msg.message_id for msg in messages],
            "timestamp": [msg.timestamp for msg in messages]
        }
    )

def merge_datasets(old_dataset: Dataset, new_dataset: Dataset) -> Dataset:
    # Gather existing URLs from old_dataset using indices if it exists
    existing_image_urls = (
        old_dataset["train"]["link"] if old_dataset is not None else []
    )

    print(f"Current rows: {len(existing_image_urls)}")

    # Filter new_dataset to only include rows that aren't in old_dataset
    filtered_new_dataset = new_dataset.filter(
        lambda example: example["link"] not in existing_image_urls
    )
    print(f"Rows to add: {filtered_new_dataset.num_rows}")

    # Concatenate the old and filtered new datasets
    if old_dataset is None:
        merged_dataset = filtered_new_dataset
    else:
        merged_dataset = concatenate_datasets(
            [old_dataset["train"], filtered_new_dataset]
        )

    print(f"New rows: {merged_dataset.num_rows}")
    return merged_dataset

def get_latest_message_id(ds: Dataset) -> str:
    try:
        return ds["train"].to_pandas()["message_id"].max()
    except:
        pass

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
        return get_user_headers()

    def _get_messages(self, after_message_id: str) -> List[Dict[str, Any]]:
        all_messages = []

        before_message_id = None

        while True:
            # Construct the URL to fetch messages from the channel
            url = self.url
            
            # Add the 'before' parameter to fetch messages before a specific message ID
            if before_message_id:
                url += f'&before={before_message_id}'

            if after_message_id:
                url += f'&after={after_message_id}'

            print(f"Fetching messages from {url}")

            response = requests.get(url, headers=self.headers)


            if response.status_code == 200:
                messages = response.json()
                
                # If there are no more messages, break out of the loop
                if not messages:
                    break
                
                
                parsed_messages: List[List[HFDatasetScheme]] = [self.parse_fn(msg) for msg in messages if self.condition_fn(msg)]
                # Flatten the list of lists
                parsed_messages = [msg for msg_list in parsed_messages for msg in msg_list]
                # Add the fetched messages to the list
                all_messages.extend(parsed_messages)

                # If the last message in the response has the same ID as the last message
                # in the previous response, break out of the loop
                if messages[-1]['id']==before_message_id:
                    break

                # Use the ID of the last message in the response as the 'before' parameter
                # for the next request to fetch the next page of messages
                before_message_id = messages[-1]['id']
            else:
                print(f"Failed to fetch messages. Status code: {response.status_code}")
                break
        
        # Ensure that there are no duplicate messages
        unique_objects = set()

        # Step 3 and 4: Iterate through the list and remove duplicates
        unique_list = []

        for obj in all_messages:
            if obj not in unique_objects:
                unique_objects.add(obj)
                unique_list.append(obj)

        return unique_list
    
    def scrape(self, fetch_all: bool=False, push_to_hub: bool=True) -> Dataset:
        try:
            current_dataset = load_dataset(self.hf_dataset_name)
            after_message_id = get_latest_message_id(current_dataset) if not fetch_all else None
        except:
            current_dataset = None
            after_message_id = None
    
        messages = self._get_messages(after_message_id=after_message_id)
        print(f"Fetched {len(messages)} messages.")

        new_dataset = prepare_dataset(messages)

        if current_dataset is not None and not fetch_all:
            ds = merge_datasets(current_dataset, new_dataset)
        else:
            ds = new_dataset

        if push_to_hub:
            ds.push_to_hub(self.hf_dataset_name, token=os.environ['HF_TOKEN'])

        return ds