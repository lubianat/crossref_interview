"""
Gets all entries in the CrossRef API with a ROR ID and saves them as a JSON Lines file.
"""

import json
from pathlib import Path
from time import sleep
import requests

base_url = "https://api.crossref.org/works?filter=has-ror-id:1&rows=1000&offset="

HERE = Path(__file__).parent

DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

OUTPUT = HERE.parent / "output"
OUTPUT.mkdir(exist_ok=True)


def get_all_ror_entries_from_api():
    offset = 0
    while True:
        print(f"Getting data from offset {offset}")
        response = requests.get(base_url + str(offset))
        data = response.json()
        if data["message"]["items"]:
            with open(DATA / f"ror_data_{offset}.jsonl", "w") as f:
                for item in data["message"]["items"]:
                    f.write(json.dumps(item) + "\n")
            offset += 1000
            sleep(1)
        else:
            break
