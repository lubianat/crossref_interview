import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import requests

import json
from datetime import datetime
from typing import List
import logging
from urllib.parse import quote
import time
import helper as helper

HERE = Path(__file__).parent
DATA = HERE / "data"
PARALLEL_PROCESSING = False
OUTPUT = HERE.parent / "output"

# Get the current month
current_month = datetime.now().strftime("%Y-%m")

# Define the cache file path
cache_file_path = DATA / f"cache_{current_month}.json"

# Load the cache from the JSON file if it exists
if cache_file_path.exists():
    with cache_file_path.open("r") as cache_file:
        cache = json.load(cache_file)
else:
    cache = {}


def save_cache():
    with cache_file_path.open("w") as cache_file:
        json.dump(cache, cache_file, indent=4, sort_keys=True)


def get_ror_api_candidates_for_affiliation(
    affiliation_name: str, local=False
) -> List[str]:
    # Check if the result is in the cache
    if affiliation_name in cache:
        return cache[affiliation_name]

    # URL-encode the affiliation name
    encoded_affiliation_name = quote(affiliation_name)

    # Query the ROR API
    if local:
        url = f"http://localhost:9292/organizations?affiliation={encoded_affiliation_name}"
    else:
        url = f"https://api.ror.org/v1/organizations?affiliation={encoded_affiliation_name}"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        # Extract relevant information from the API response
        result_ror_ids = [
            {
                "id": match["organization"]["id"],
                "matching_type": match["matching_type"],
                "score": match["score"],
            }
            for match in data["items"]
        ]

    else:
        logging.error(
            f"Failed to fetch data from ROR API for {affiliation_name}: {response.status_code}"
        )
        result_ror_ids = []

    # Store the result in the cache
    cache[affiliation_name] = result_ror_ids
    save_cache()
    time.sleep(0.5)
    return result_ror_ids


def main():
    crossref_df = pd.read_csv(
        DATA.joinpath("crossref_ror_ids_with_affiliation_strings.csv")
    )
    ror_registry_df = pd.read_csv(DATA.joinpath("ror_registry_slim.csv"))
    ror_registry_dict = ror_registry_df.set_index("id").to_dict(orient="index")

    wikidata_ror_dataset = pd.read_csv(DATA.joinpath("wikidata_ror_dataset.csv"))
    wikidata_ror_dict = wikidata_ror_dataset.set_index("ROR_ID").to_dict(orient="index")

    def detect_lexical_match(row, ror_registry_dict, wikidata_ror_dict):
        ror_id = row["ROR_ID"]
        affiliation_name = row["normalized_name"]

        if ror_id not in ror_registry_dict:
            return "ror not registered"

        ror_data = ror_registry_dict[ror_id]
        ror_id_acronyms = extract_clean_labels(
            ror_data["names.types.acronym.normalized"]
        )
        ror_id_aliases = extract_clean_labels(ror_data["names.types.alias.normalized"])
        ror_id_labels = extract_clean_labels(ror_data["names.types.label.normalized"])
        ror_id_ror_display = ror_data["names.types.ror_display.normalized"]

        match_type = "no match"
        if affiliation_name == ror_id_ror_display:
            match_type = "ror_display"
        elif affiliation_name in ror_id_labels:
            match_type = "label"
        elif affiliation_name in ror_id_aliases:
            match_type = "alias"
        elif affiliation_name in ror_id_acronyms:
            match_type = "acronym"

        elif "," in affiliation_name:
            affiliation_name_parts = affiliation_name.split(", ")
            affiliation_name_parts = [name.strip() for name in affiliation_name_parts]
            if any(part in ror_id_labels for part in affiliation_name_parts):
                match_type = "comma-separated subpart label match"
            elif any(part in ror_id_aliases for part in affiliation_name_parts):
                match_type = "comma-separated subpart alias match"
            elif any(part in ror_id_acronyms for part in affiliation_name_parts):
                match_type = "comma-separated subpart acronym match"

        if "(" in affiliation_name and ")" in affiliation_name:
            inside_the_parenthesis = (
                affiliation_name.split("(")[1].split(")")[0].strip()
            )
            before_the_parenthesis = affiliation_name.split("(")[0].strip()
            if inside_the_parenthesis in ror_id_labels:
                match_type = "inner parenthesis label match"
            elif inside_the_parenthesis in ror_id_aliases:
                match_type = "inner parenthesis alias match"
            elif inside_the_parenthesis in ror_id_acronyms:
                match_type = "inner parenthesis acronym match"
            elif before_the_parenthesis in ror_id_labels:
                match_type = "before parenthesis label match"
            elif before_the_parenthesis in ror_id_aliases:
                match_type = "before parenthesis alias match"
            elif before_the_parenthesis in ror_id_acronyms:
                match_type = "before parenthesis acronym match"

        if match_type == "no match":
            # Space separated name parts =
            space_name_parts = affiliation_name.split(" ")
            space_name_parts = [name.strip() for name in space_name_parts]
            if any(part in ror_id_labels for part in space_name_parts):
                match_type = "space-separated subpart label match"
            elif any(part in ror_id_aliases for part in space_name_parts):
                match_type = "space-separated subpart alias match"
            elif any(part in ror_id_acronyms for part in space_name_parts):
                match_type = "space-separated subpart acronym match"

        if match_type == "no match":
            wikidata_ror_data = wikidata_ror_dict.get(ror_id, "")
            if wikidata_ror_data:
                wikidata_labels = wikidata_ror_data.get("LABEL", "").split(" | ")
                wikidata_labels = [label.strip() for label in wikidata_labels]
                if affiliation_name in wikidata_labels:
                    match_type = "wikidata"
                elif "," in affiliation_name:
                    affiliation_name_parts = affiliation_name.split(", ")
                    if any(part in wikidata_labels for part in affiliation_name_parts):
                        match_type = "comma-separated subpart wikidata match"
            # Space-separated acronym match

        if match_type == "no match":
            if ror_id_ror_display in affiliation_name:
                match_type = "ror_display as substring"
            elif any(ror_label in affiliation_name for ror_label in ror_id_labels):
                match_type = "label as substring"
            elif any(ror_alias in affiliation_name for ror_alias in ror_id_aliases):
                match_type = "alias as substring"

        if match_type == "no match":
            if affiliation_name in ror_id_ror_display:
                match_type = "name as substring of ror_display"
            elif any(affiliation_name in ror_label for ror_label in ror_id_labels):
                match_type = "name as substring of label"
            elif any(affiliation_name in ror_alias for ror_alias in ror_id_aliases):
                match_type = "name as substring of alias"

        if match_type == "no match":
            ror_candidates = get_ror_api_candidates_for_affiliation(
                affiliation_name=affiliation_name, local=False
            )
            ror_candidate_ids = [candidate["id"] for candidate in ror_candidates]
            ror_candidate_match_types = [
                candidate["matching_type"] for candidate in ror_candidates
            ]
            if ror_id in ror_candidate_ids:
                ror_id_position_among_candidates = ror_candidate_ids.index(ror_id)
                api_match_type = ror_candidate_match_types[
                    ror_id_position_among_candidates
                ]
                match_type = (
                    "ror affiliation endpoint match position "
                    + str(ror_id_position_among_candidates + 1)
                    + f" ({api_match_type})"
                )

        if match_type == "no match":
            # check status for ror status
            ror_status = ror_data.get("status", "")
            if ror_status == "withdrawn":
                match_type = "no match (withdrawn ror)"

        ror_to_medical_school_mappings = {
            "https://ror.org/03vek6s52": [
                "Harvard T.H. Chan School of Public Health",
                "harvard t h chan school of public health",
                "harvard th chan school of public health",
                "Harvard Medical School",
            ],
            "https://ror.org/01yc7t268": ["Washington University School of Medicine"],
            "https://ror.org/036c27j91": [
                "washington university school of medicine"
            ],  # Also weird, two rors, same string
            "https://ror.org/0107w4315": [
                "University of Colorado Medical School"
            ],  # Actually ROR for  University of Colorado Health
            "https://ror.org/03v76x132": [
                "Yale School of Medicine",
                "yale school of public health",
            ],
            "https://ror.org/04rq5mt64": ["university of maryland school of medicine"],
            "https://ror.org/03wmf1y16": ["university of colorado school of medicine"],
            "https://ror.org/0202bj006": [
                "shengjing hospital of china medical university"
            ],  # ROR for parent hostpital
            "https://ror.org/03gds6c39": ["mcgovern medical school"],
            "https://ror.org/04vmvtb21": ["tulane school of medicine"],
            "https://ror.org/049s0rh22": ["geisel school of medicine at dartmouth"],
            "https://ror.org/00jmfr291": ["university of michigan medical school"],
        }

        # normalize ror_to_medical_school_mappings using helper.normalize_strings
        ror_to_medical_school_mappings = {
            helper.normalize_string(k): [helper.normalize_string(v) for v in vs]
            for k, vs in ror_to_medical_school_mappings.items()
        }

        if match_type == "no match":
            medical_schools = ror_to_medical_school_mappings.get(ror_id, [])
            if len(medical_schools) != 0:

                if affiliation_name in medical_schools:
                    match_type = "medical school to university mapping"
                elif any(
                    medical_school in affiliation_name
                    for medical_school in medical_schools
                ):
                    match_type = "medical school to university mapping (substring)"

        if match_type == "no match" and ror_id == "https://ror.org/04qw24q55":
            match_type = "no match (special case, Wageningen University)"

        return match_type

    tqdm.pandas()
    crossref_df["match_type"] = crossref_df.progress_apply(
        detect_lexical_match,
        axis=1,
        ror_registry_dict=ror_registry_dict,
        wikidata_ror_dict=wikidata_ror_dict,
    )

    crossref_df.to_csv(
        DATA.joinpath("crossref_ror_ids_with_affiliation_strings_and_matches.tsv"),
        sep="\t",
        index=False,
    )

    # Filter df by no-match and save
    no_match_df = crossref_df[crossref_df["match_type"] == "no match"]
    no_match_df.to_csv(
        DATA.joinpath("crossref_ror_ids_with_affiliation_strings_and_no_matches.tsv"),
        sep="\t",
        index=False,
    )
    # Overall match summary
    match_summary = crossref_df["match_type"].value_counts().reset_index()
    match_summary.columns = ["match_type", "count"]
    match_summary_html = match_summary.to_html(index=False)

    # Count the number of "no-match" per ror, show ordered list of ror_ids with the most no-matches
    # Include ror_display names, links, prepare for use datatable.js

    no_match_counts = no_match_df["ROR_ID"].value_counts().reset_index()
    no_match_counts.columns = ["ROR_ID", "no_match_count"]
    no_match_counts = no_match_counts.merge(
        ror_registry_df[["id", "names.types.ror_display"]],
        left_on="ROR_ID",
        right_on="id",
    )
    no_match_counts = no_match_counts.sort_values("no_match_count", ascending=False)

    # Add a new column showing the unique strings provided as "name" in crossref
    unique_names = (
        no_match_df.groupby("ROR_ID")["normalized_name"].unique().reset_index()
    )
    unique_names.columns = ["ROR_ID", "unique_names"]
    no_match_counts = no_match_counts.merge(unique_names, on="ROR_ID")

    # No, add a new column showing the unique doi prefixes (before first /) for each un-matched ROR.

    unique_prefixes = (
        no_match_df["DOI"]
        .apply(lambda x: x.split("/")[0])
        .groupby(no_match_df["ROR_ID"])
        .unique()
        .reset_index()
    )
    unique_prefixes.columns = ["ROR_ID", "unique_prefixes"]
    # no_match_counts = no_match_counts.merge(unique_prefixes, on="ROR_ID")

    # Now, for each prefix, calculate the number of envolved unique DOIs.
    # Instead of a list, display it as a dict of "prefix": count
    # Create a DataFrame with prefixes and ROR_ID

    prefixes_df = no_match_df.assign(
        prefix=no_match_df["DOI"].apply(lambda x: x.split("/")[0])
    )

    # Group by ROR_ID and prefix, then count unique DOIs
    prefix_counts = (
        prefixes_df.groupby(["ROR_ID", "prefix"])["DOI"]
        .nunique()
        .reset_index()
        .groupby("ROR_ID")
        .apply(lambda x: dict(zip(x["prefix"], x["DOI"])))
        .reset_index()
        .rename(columns={0: "prefix_counts"})
    )

    # Merge the result back to no_match_counts
    no_match_counts = no_match_counts.merge(prefix_counts, on="ROR_ID")

    no_match_count_summary_html = no_match_counts.to_html(index=False)

    # Create an HTML file showing the decision tree for the matching process
    decision_tree_html = """
    <h2>Decision Tree for ROR Matching</h2>
    <p>Matching is done in the following order:</p>
    <ol>
        <li>Exact match with ROR display name</li>
        <li>Exact match with ROR label</li>
        <li>Exact match with ROR alias</li>
        <li>Exact match with ROR acronym</li>
        <li>Comma-separated subpart match with ROR label</li>
        <li>Comma-separated subpart match with ROR alias</li>
        <li>Comma-separated subpart match with ROR acronym</li>
        <li>Space-separated supart match with ROR label </li>
        <li>Space-separated supart match with ROR alias </li>
        <li>Space-separated supart match with ROR acronym </li>
        <li>Exact match with Wikidata label</li>
        <li>Comma-separated subpart match with Wikidata label</li>
    </ol>
    """
    # GENERATE HTML combining the above plots and tables
    html_parts = []
    html_parts.append(decision_tree_html)
    html_parts.append("<h2>Match Summary</h2>")
    html_parts.append(match_summary_html)
    html_parts.append("<h2>No Match Counts</h2>")
    html_parts.append(no_match_count_summary_html)
    html_output = "\n".join(html_parts)
    with open(OUTPUT.joinpath("ror_matching_analysis.html"), "w") as f:
        f.write(html_output)


def extract_clean_labels(ror_id_labels):
    if pd.isna(ror_id_labels):
        return []
    clean_labels = [
        label.split(": ")[1] if ": " in label else label
        for label in ror_id_labels.split("; ")
    ]
    clean_labels = [label.lower() for label in clean_labels]
    clean_labels = [label.strip() for label in clean_labels]
    return clean_labels


if __name__ == "__main__":
    main()
