import os
import json
from typing import List, Tuple, Dict, Any
from datasets import Dataset, load_dataset

import utils
config = utils.load_config()

if __name__ == "__main__":
    with open("config.json", "r") as f:
        local_config = json.load(f)
    print(f"Backing up {config['hf_dataset_name']} to {local_config['hf_dataset_name']}...")
    current_dataset = load_dataset(config["hf_dataset_name"])
    print(f"Loaded current dataset: {current_dataset} from {config['hf_dataset_name']}")
    current_dataset.push_to_hub(local_config["hf_dataset_name"], token=os.environ["HF_TOKEN"])
    print(f"Pushed current dataset to {local_config['hf_dataset_name']}")
