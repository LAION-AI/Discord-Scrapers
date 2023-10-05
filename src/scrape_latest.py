import requests
from typing import List, Dict, Tuple

import utils

def get_latest_messages() -> List[Tuple[str, str]]:
    config = utils.load_config()

    base_url = config['base_url']
    limit = config['limit']
    channel_id = config["channel_id"]

    headers = utils.get_user_headers()
    # Construct the URL to fetch messages from the channel
    url = f'{base_url}/channels/{channel_id}/messages?limit={limit}'

    response = requests.get(url, headers=headers)

    latest_messages = []

    if response.status_code == 200:
        messages = response.json()
        for message in messages:
            # Assumes message has the following structure:
            # "<prompt>"
            # <image attachment>
            if len(message["attachments"]) == 0 or not message["content"].startswith('"'):
                continue
            prompt_image = utils.parse_message(message)
            latest_messages.append(prompt_image)
    else:
        print(f"Failed to fetch messages. Status code: {response.status_code}")

    return latest_messages

if __name__ == '__main__':
    messages = get_latest_messages()
    print(f"Fetched {len(messages)} messages.")
    print(messages)