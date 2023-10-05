import os
import json
from typing import List, Tuple, Dict, Any

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
