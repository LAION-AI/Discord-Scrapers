import requests
from typing import List, Dict, Tuple

from datasets import Dataset, load_dataset

import utils
import os

def get_all_messages() -> List[Tuple[str, str]]:
    config = utils.load_config()

    base_url = config['base_url']
    limit = config['limit']
    channel_id = config["channel_id"]

    headers = utils.get_user_headers()

    all_messages = []

    # Initialize with a large limit to get as many messages as possible per request
    limit = 100
    before_message_id = None

    while True:
        # Construct the URL to fetch messages from the channel
        url = f'{base_url}/channels/{channel_id}/messages?limit={limit}'
        
        # Add the 'before' parameter to fetch messages before a specific message ID
        if before_message_id:
            url += f'&before={before_message_id}'

        print(f"Fetching messages from {url}...")

        response = requests.get(url, headers=headers)


        if response.status_code == 200:
            messages = response.json()
            
            # If there are no more messages, break out of the loop
            if not messages:
                break
            
            # Add the fetched messages to the list
            all_messages.extend(messages)

            # Use the ID of the last message in the response as the 'before' parameter
            # for the next request to fetch the next page of messages
            before_message_id = messages[-1]['id']
            limit = min(limit, 1000)  # Reduce the limit to 1000 for subsequent requests
        else:
            print(f"Failed to fetch messages. Status code: {response.status_code}")
            break

    messages = []
    for message in all_messages:
        # Assumes message has the following structure:
        # "<prompt>"
        # <image attachment>
        if len(message["attachments"]) == 0 or not message["content"].startswith('"'):
            continue
        prompt_images = utils.parse_message(message)
        messages.extend(prompt_images)

    return messages

if __name__ == '__main__':
    messages = get_all_messages()
    dataset = utils.prepare_dataset(messages)
    print(f"Fetched {len(messages)} messages.")
    current_dataset = load_dataset("ZachNagengast/LAION-discord-dalle3", token=os.environ['HF_TOKEN'])
    print(dataset)
    merged_dataset = utils.merge_datasets(current_dataset, dataset)
    merged_dataset.push_to_hub("ZachNagengast/LAION-discord-dalle3", token=os.environ['HF_TOKEN'])