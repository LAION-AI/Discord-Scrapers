import io
import os
import pandas as pd
from typing import Any, Dict, List
import requests
from PIL import Image as PILImage
from scraper import ScraperBot, ScraperBotConfig
from dataclasses import dataclass
from datasets import Image


@dataclass(frozen=True)
class HFDatasetScheme:
    caption: str
    image: Image(decode=True)
    link: str
    message_id: str
    timestamp: str


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
    prompt = message["content"].split("- <@")[0].strip().replace("**", "")
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
    return len(message["attachments"]) > 0


def prepare_dataset(messages: List[HFDatasetScheme]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caption": [msg.caption for msg in messages],
            "image": [
                None for msg in messages
            ],  # Initialize to None, will be filled in later
            "link": [
                msg.link for msg in messages
            ],  # will maintain just because we use it to filter
            "message_id": [msg.message_id for msg in messages],
            "timestamp": [msg.timestamp for msg in messages],
        }
    )


def get_image(link: str) -> bytes:
    image = PILImage.open(requests.get(link, stream=True).raw).convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return {"bytes": img_byte_arr.getvalue(), "path": None}


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = ScraperBotConfig.from_json(config_path)

    bot = ScraperBot(config=config, HFDatasetScheme=HFDatasetScheme, prepare_dataset=prepare_dataset, parse_fn=parse_fn, condition_fn=condition_fn, download_fn=get_image)
    bot.scrape(fetch_all=os.environ.get("FETCH_ALL", "false").lower() == "true")
