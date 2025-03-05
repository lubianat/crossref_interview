#!/usr/bin/env python3
import json
import csv
import os
import gzip
import logging
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_records_from_file(file_path: str) -> Optional[List[Dict]]:
    try:
        with gzip.open(file_path, mode="rt", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "items" in data and isinstance(data["items"], list):
                    return data["items"]
                else:
                    return [data]
            elif isinstance(data, list):
                return data
            else:
                logging.warning(f"Unexpected JSON structure in file: {file_path}")
                return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
    return None


def process_record(record: Dict) -> List[List[str]]:
    rows = []
    doi = record.get("DOI")
    if not doi:
        return rows

    authors = record.get("author")
    if not authors or not isinstance(authors, list):
        return rows

    for author in authors:
        affiliations = author.get("affiliation")
        if not affiliations or not isinstance(affiliations, list):
            continue

        for aff in affiliations:
            affiliation_name = aff.get("name", "").strip()
            ids = aff.get("id")
            if ids and isinstance(ids, list):
                for aff_id in ids:
                    ror_id = aff_id.get("id", "").strip()
                    if ror_id:
                        rows.append([doi, affiliation_name, ror_id])
            else:
                ror_id = aff.get("ror") or aff.get("ror-id")
                if ror_id:
                    rows.append([doi, affiliation_name, ror_id])
    return rows


def process_single_file(file_path: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Process a single file and return its filename along with the list of extracted rows.
    Rows are returned as tuples to facilitate deduplication.
    """
    rows = []
    records = extract_records_from_file(file_path)
    if records is None:
        return (os.path.basename(file_path), rows)
    for record in records:
        for row in process_record(record):
            # Convert row to tuple for easier deduplication later
            rows.append(tuple(row))
    return (os.path.basename(file_path), rows)


def process_data_directory_parallel(
    data_dir: str, writer: csv.writer, csvfile, processed_files_log: str
):
    # Load the list of processed files
    processed_files = set()
    if os.path.exists(processed_files_log):
        with open(processed_files_log, "r", encoding="utf-8") as pf:
            for line in pf:
                processed_files.add(line.strip())

    # Get list of files to process
    file_list = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".json.gz") and filename not in processed_files
    ]

    seen_rows = set()  # for deduplication

    with ProcessPoolExecutor(max_workers=10) as executor:
        # Submit a processing task for each file
        future_to_file = {
            executor.submit(process_single_file, file_path): file_path
            for file_path in file_list
        }

        # Use tqdm to monitor progress
        for future in tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc="Processing files",
        ):
            file_path = future_to_file[future]
            try:
                filename, file_rows = future.result()
            except Exception as exc:
                logging.error(
                    f"File {os.path.basename(file_path)} generated an exception: {exc}"
                )
                continue

            # Write unique rows to the CSV file
            new_rows = 0
            for row in file_rows:
                if row not in seen_rows:
                    writer.writerow(row)
                    seen_rows.add(row)
                    new_rows += 1

            # Log file as processed
            with open(processed_files_log, "a", encoding="utf-8") as pf:
                pf.write(filename + "\n")

            # Flush CSV output immediately
            csvfile.flush()
            os.fsync(csvfile.fileno())

            logging.info(
                f"Processed {filename}: {len(file_rows)} rows, {new_rows} new unique rows."
            )


def main():
    setup_logging()
    logging.info("Starting Crossref ROR extraction process with parallel processing.")

    data_dir = DATA / "crossref_raw_data"
    output_csv = "crossref_affiliation_ids.csv"
    processed_files_log = DATA / "crossref_processed_files.txt"

    if not os.path.isdir(data_dir):
        logging.error(f"Data directory does not exist: {data_dir}")
        return

    try:
        # Open CSV in write mode (if resuming, consider append mode and checking header)
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["DOI", "Affiliation_Name", "ROR_ID"])
            process_data_directory_parallel(
                data_dir, writer, csvfile, processed_files_log
            )
    except Exception as e:
        logging.error(f"Error writing output CSV: {e}")
        return

    logging.info(f"Processing complete. Extracted data saved to {output_csv}")


if __name__ == "__main__":
    main()
