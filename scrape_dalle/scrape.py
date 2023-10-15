import os
from typing import Any, Dict, List

import sys
sys.path.append("..")

from scraper import ScraperBot, ScraperBotConfig, HFDatasetScheme


def parse_fn(message: Dict[str, Any]) -> List[HFDatasetScheme]:
    """Parses a message into a list of Hugging Face Dataset Schemes.

    Parameters
    ----------
    message : Dict[str, Any]
        The message to parse.

    Returns
    -------
    List[HFDatasetScheme]
        A list of Hugging Face Dataset Schemes.
    """
    content = message["content"]

    # Find the index of the first quote in the content
    first_quote_index = content.find('"')

    # Find the index of the last quote in the content
    last_quote_index = content.rfind('"')

    # Extract the text between the first and last quotes to get the complete prompt
    prompt = content[first_quote_index + 1:last_quote_index].strip()
    image_urls = [attachment["url"] for attachment in message["attachments"]]
    timestamp = message["timestamp"]
    message_id = message["id"]

    return [HFDatasetScheme(caption=prompt, image=None, link=image_url, message_id=message_id, timestamp=timestamp)
            for image_url in image_urls]


def condition_fn(message: Dict[str, Any]) -> bool:
    """Checks if a message meets the condition to be parsed.

    Parameters
    ----------
    message : Dict[str, Any]
        The message to check.

    Returns
    -------
    bool
        True if the message meets the condition, False otherwise.
    """
    return len(message["attachments"]) > 0 and message["content"].startswith('"')


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = ScraperBotConfig.from_json(config_path)

    bot = ScraperBot(config=config, parse_fn=parse_fn, condition_fn=condition_fn)
    bot.scrape(fetch_all=os.environ.get("FETCH_ALL", "false").lower() == "true")
