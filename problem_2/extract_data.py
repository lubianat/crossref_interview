#!/usr/bin/env python3
import json
import csv
import os
import gzip
import logging
from typing import Optional, List, Dict
from tqdm import tqdm

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def extract_records_from_file(file_path: str) -> Optional[List[Dict]]:
    """
    Extract records from a gzipped JSON file.
    
    The JSON content might be structured as:
      - A dictionary with an "items" key (list of records)
      - A list of records
      - A single record (dictionary)
    
    Returns:
      A list of record dictionaries, or None if an error occurs.
    """
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
    """
    Process a single record to extract rows of data.
    
    For each record, if a DOI exists and the record contains an 'author' field,
    this function iterates over authors and their affiliations to find any ROR IDs.
    
    Returns:
      A list of rows, where each row is [DOI, Affiliation_Name, ROR_ID].
    """
    rows = []
    doi = record.get("DOI")
    if not doi:
        return rows  # Skip if DOI is missing

    authors = record.get("author")
    if not authors or not isinstance(authors, list):
        return rows

    for author in authors:
        affiliations = author.get("affiliation")
        if not affiliations or not isinstance(affiliations, list):
            continue

        for aff in affiliations:
            affiliation_name = aff.get("name", "").strip()
            # Look for ROR IDs stored in a list under key "id"
            ids = aff.get("id")
            if ids and isinstance(ids, list):
                for aff_id in ids:
                    ror_id = aff_id.get("id", "").strip()
                    if ror_id:
                        rows.append([doi, affiliation_name, ror_id])
            # Alternatively, check for ROR IDs under "ror" or "ror-id"
            else:
                ror_id = aff.get("ror") or aff.get("ror-id")
                if ror_id:
                    rows.append([doi, affiliation_name, ror_id])
    return rows

def process_data_directory(data_dir: str, writer: csv.writer):
    """
    Process all gzipped JSON files in the specified directory.
    
    This function iterates over each file, extracts the records, processes each record,
    and writes out unique rows (if any) to the CSV writer.
    """
    file_count = 0
    record_count = 0
    output_row_count = 0
    seen_rows = set()  # Using a set for deduplication; ~1M rows is acceptable in-memory on your machine.

    # Use tqdm to iterate over files for progress tracking
    for filename in tqdm(os.listdir(data_dir), desc="Processing files"):
        if filename.endswith(".json.gz"):
            file_path = os.path.join(data_dir, filename)
            logging.info(f"Processing file: {filename}")
            records = extract_records_from_file(file_path)
            if records is None:
                logging.warning(f"Skipping file due to extraction issues: {filename}")
                continue

            file_count += 1
            # Use tqdm to track records processing per file
            for record in tqdm(records, desc=f"Processing records in {filename}", leave=False):
                record_count += 1
                rows = process_record(record)
                for row in rows:
                    row_tuple = tuple(row)
                    if row_tuple not in seen_rows:
                        writer.writerow(row)
                        seen_rows.add(row_tuple)
                        output_row_count += 1
            logging.info(f"File {filename}: Processed {len(records)} records; total unique rows so far: {output_row_count}")

    logging.info(f"Total: Processed {file_count} files, {record_count} records, and extracted {output_row_count} unique rows.")

def main():
    setup_logging()
    logging.info("Starting Crossref ROR extraction process.")

    # Define paths (adjust these as needed)
    data_dir = "/home/lubianat/Documents/random/crossref_data_exploration/problem_2/data"
    output_csv = "crossref_ror_extracted.csv"

    if not os.path.isdir(data_dir):
        logging.error(f"Data directory does not exist: {data_dir}")
        return

    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["DOI", "Affiliation_Name", "ROR_ID"])
            process_data_directory(data_dir, writer)
    except Exception as e:
        logging.error(f"Error writing output CSV: {e}")
        return

    logging.info(f"Processing complete. Extracted data saved to {output_csv}")

if __name__ == "__main__":
    main()
