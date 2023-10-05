import os
import json
from typing import List, Tuple, Dict, Any
from datasets import Dataset, concatenate_datasets

def get_bot_headers() -> Dict[str, str]:
    return {
        'Authorization': f"Bot {os.environ['DISCORD_TOKEN']}"
    }

def get_user_headers() -> Dict[str, str]:
    return {
        'authorization': os.environ['DISCORD_TOKEN']
    }

def load_config() -> Dict[str, Any]:
    with open('config.json', 'r') as f:
        return json.load(f)
    
def parse_message(message: Dict[str, Any]) -> List[Tuple[str, str]]:
    # Assumes that message has the following format:
    # "<prompt>"
    # <image attachment>
    prompt = message["content"].split('"')[1].strip()
    image_urls = [attachment["url"] for attachment in message["attachments"]]

    return [(prompt, image_url) for image_url in image_urls]

def prepare_dataset(messages: List[Tuple[str, str]]) -> Dataset:
    dataset = Dataset.from_dict({
        "caption": [prompt for prompt, _ in messages],
        "link": [image_url for _, image_url in messages]
    })
    return dataset

def merge_datasets(old_dataset: Dataset, new_dataset: Dataset) -> Dataset:
    # Gather existing URLs from old_dataset using indices
    existing_image_urls = old_dataset['train']['link']

    # Filter new_dataset to only include rows that aren't in old_dataset
    filtered_new_dataset = new_dataset.filter(lambda example: example['link'] not in existing_image_urls)
    print(filtered_new_dataset)
    
    # Concatenate the old and filtered new datasets
    merged_dataset = concatenate_datasets([old_dataset['train'], filtered_new_dataset])

    print(merged_dataset)
    return merged_dataset


    