import json
import time
import requests
from pathlib import Path

base_url = "https://api.crossref.org/works?filter=has-ror-id:1&rows=1000"

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

OUTPUT = HERE.parent / "output"
OUTPUT.mkdir(exist_ok=True)


def get_all_ror_entries_from_api():
    cursor = "*"
    # Path for storing the downloaded data:
    out_file = DATA / "crossref_ror_entries_2023-10-03.jsonl"

    # Open the file once, in write mode, for consistent output:
    with open(out_file, "w", encoding="utf-8") as f:
        while True:
            url = f"{base_url}&cursor={cursor}"
            print(f"Fetching: {url}")

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                # Handle network errors or rate-limiting (HTTP 4xx/5xx)
                print(f"Request failed: {e}. Retrying in 60 seconds...")
                time.sleep(60)
                continue

            data = response.json()
            message = data.get("message", {})
            items = message.get("items", [])

            # Write each item to the file as a separate JSON line:
            for item in items:
                f.write(json.dumps(item) + "\n")

            # Get the next cursor:
            next_cursor = message.get("next-cursor")

            # If there's no next cursor, or if it hasn't changed, we're done:
            if not next_cursor or next_cursor == cursor:
                print("No more results, or cursor is not changing. Done.")
                break

            # Update cursor and pause to avoid hitting rate limits:
            cursor = next_cursor
            time.sleep(1)  # Sleep 1 second between requests

    print(f"Data collection complete. See {out_file} for results.")


if __name__ == "__main__":
    get_all_ror_entries_from_api()
