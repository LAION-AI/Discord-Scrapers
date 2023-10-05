import os
import json
from typing import List, Tuple, Dict, Any
from datasets import Dataset, load_dataset, concatenate_datasets

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

def prepare_dataset(messages: List[Tuple[str, str]], config) -> Dataset:
    dataset = Dataset.from_dict({
        "caption": [prompt for prompt, _ in messages],
        "link": [image_url for _, image_url in messages]
    })

    current_dataset = load_dataset(config['hf_dataset_name'])
    merged_dataset = merge_datasets(current_dataset, dataset)

    return merged_dataset

def merge_datasets(old_dataset: Dataset, new_dataset: Dataset) -> Dataset:
    # Gather existing URLs from old_dataset using indices
    existing_image_urls = old_dataset['train']['link']

    print(f"Current rows: {len(existing_image_urls)}")

    # Filter new_dataset to only include rows that aren't in old_dataset
    filtered_new_dataset = new_dataset.filter(lambda example: example['link'] not in existing_image_urls)
    print(f"Rows to add: {filtered_new_dataset.num_rows}")
    
    # Concatenate the old and filtered new datasets
    merged_dataset = concatenate_datasets([old_dataset['train'], filtered_new_dataset])

    print(f"New rows: {merged_dataset.num_rows}")
    return merged_dataset

def upload_dataset(dataset: Dataset, config) -> None:
    dataset.push_to_hub(config['hf_dataset_name'], token=os.environ['HF_TOKEN'])


    