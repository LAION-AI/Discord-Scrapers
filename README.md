# Discord-Scraper

Pipeline to scrape prompt + image url pairs from Discord channels. The idea started by wanting to scrape the image-prompt pairs from [share-dalle-3](https://discord.com/channels/823813159592001537/1158354590463447092) Discord channel from `LAION` server. But now you can re-use the scraper to work with any channel you want.

## How to use

Clone the repo `git clone https://github.com/LAION-AI/Discord-Scrapers.git`

1. Set up a virtual environment and install the requirements with `pip install -r requirements.txt`
2. Get your `DISCORD_TOKEN` and `HF_TOKEN` and add as environment variables. 
   1. `DISCORD_TOKEN` can be obtained by looking at developer tools in your Web Browser
   2. `HF_TOKEN` can be obtained by logging in to HuggingFace and looking at your profile
3. Get the `channel_id` from the Discord channel you want to scrape. You can do this by enabling developer mode in Discord and right clicking the channel you want to scrape.
4. Create a `condition_fn` and a `parse_fn` that will be used to filter and parse the messages. You can use the ones I created as an example.
5. Create your scraping script and optionally your `config.json`


**NOTE PAY ATTENTION TO THE FUNC SIGNATURE OF parse_fn and condition_fn**

```python
import os
from typing import Any, Dict, List

from scraper import ScraperBot, ScraperBotConfig, HFDatasetScheme

def parse_fn(message: Dict[str, Any]) -> List[HFDatasetScheme]:
    ...

def condition_fn(message: Dict[str, Any]) -> bool:
    ...

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = ScraperBotConfig.from_json(config_path)

    bot = ScraperBot(config=config, parse_fn=parse_fn, condition_fn=condition_fn)
    bot.scrape(fetch_all=False, push_to_hub=False)
```


## Main Components

### ScraperBotConfig

Dataclass with configuration attributes to be used by the ScraperBot. You can create your own config.json file and load it with `ScraperBotConfig.from_json(path_to_config)`.

attributes:
- base_url: str, The base url of the Discord API (in chase it changes)
- channel_id: str, The id of the channel you want to scrape
- limit: int, The number of messages to fetch (from my tests the max allowed by Discord is 100)
- hf_dataset_name: str, The name of the dataset you want to push to HuggingFace

### ScraperBot

Implementation of the scraper. Get's the messages from the Discord API and filters them using the `condition_fn`. Then parses the messages using the `parse_fn` and pushes the dataset to HuggingFace.

attributes:
- config: ScraperBotConfig, The configuration to be used by the bot
- parse_fn: Callable[[Dict[str, Any]], List[HFDatasetScheme]], The function to parse the messages
- condition_fn: Callable[[Dict[str, Any]], bool], The function to filter the messages

methods:

#### scrape(fetch_all: bool = False, push_to_hub: bool = False) -> Dataset 

Scrapes the messages and optionally pushes the dataset to HuggingFace.

args:
- fetch_all: bool, If True will fetch all the messages from the channel. If False will fetch only the messages that weren't processed yet.
- push_to_hub: bool, If True will push the dataset to HuggingFace. If False will only return the dataset.

**NOTE: If you want to push the dataset to HuggingFace you need to set the `HF_TOKEN` environment variable.**
**NOTE 2: If the dataset doesn't exist in HuggingFace it will be created. If it already exists it will be updated.**
  