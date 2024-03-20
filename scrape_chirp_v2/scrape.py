import io
import os
import re
import pandas as pd
from typing import Any, Dict, List
import requests
from PIL import Image as PILImage
from scraper import ScraperBot, ScraperBotConfig
from helpers import has_code_block, get_code_block
from dataclasses import dataclass
from datasets import Audio


@dataclass(frozen=True)
class HFDatasetScheme:
    user_prompt: str
    system_prompt: str
    lyrics: str
    audio: Audio(decode=True)
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
    content = message["content"]

    prompt = get_code_block(content).strip()

    """
    The code block looks like this:
    USER_PROMPT:A 90s alt rock song about being lazy and crazy
    SYSTEM_PROMPT:alternative rock grungy energetic
    LYRICS:[Verse] (many lines of lyrics)
    """

    user_prompt = prompt.split("\n")[0].split(":")[1].strip()
    system_prompt = prompt.split("\n")[1].split(":")[1].strip()
    lyrics = "\n".join(prompt.split("\n")[2:]).strip()

    # Extract the text between the first and last quotes to get the complete prompt
    audio_urls = [attachment["url"] for attachment in message["attachments"]]
    timestamp = message["timestamp"]
    message_id = message["id"]

    return [
        HFDatasetScheme(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            lyrics=lyrics,
            audio=None,
            link=audio_url,
            message_id=message_id,
            timestamp=timestamp,
        )
        for audio_url in audio_urls
    ]


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
    return has_code_block(message["content"]) and len(message["attachments"]) > 0


def prepare_dataset(messages: List[HFDatasetScheme]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_prompt": [msg.user_prompt for msg in messages],
            "system_prompt": [msg.system_prompt for msg in messages],
            "lyrics": [msg.lyrics for msg in messages],
            "audio": [
                None for msg in messages
            ],  # Initialize to None, will be filled in later
            "link": [
                msg.link for msg in messages
            ],  # will maintain just because we use it to filter
            "message_id": [msg.message_id for msg in messages],
            "timestamp": [msg.timestamp for msg in messages],
        }
    )


def download_fn(link: str) -> bytes:
    response = requests.get(link)
    return response.content


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = ScraperBotConfig.from_json(config_path)

    bot = ScraperBot(
        config=config,
        HFDatasetScheme=HFDatasetScheme,
        prepare_dataset=prepare_dataset,
        parse_fn=parse_fn,
        condition_fn=condition_fn,
        download_fn=download_fn,
        readme_template="dataset_readme_template_chirp.md",
    )
    bot.scrape(fetch_all=os.environ.get("FETCH_ALL", "false").lower() == "true")
