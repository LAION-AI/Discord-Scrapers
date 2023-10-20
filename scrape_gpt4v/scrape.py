import os
import re
from typing import Any, Dict, List

from scraper import ScraperBot, ScraperBotConfig, HFDatasetScheme
from helpers import starts_with_quotes, get_start_end_quotes

url_pattern = re.compile(r'https?://\S+')

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

    (first_quote_index, last_quote_index) = get_start_end_quotes(content)

    # Extract the text between the first and last quotes to get the complete prompt
    prompt = content[first_quote_index + 1:last_quote_index].strip()
    image_urls = url_pattern.findall(content)
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
    return url_pattern.search(message["content"]) and starts_with_quotes(message["content"])


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = ScraperBotConfig.from_json(config_path)

    bot = ScraperBot(config=config, parse_fn=parse_fn, condition_fn=condition_fn)
    bot.scrape(fetch_all=os.environ.get("FETCH_ALL", "false").lower() == "true")
