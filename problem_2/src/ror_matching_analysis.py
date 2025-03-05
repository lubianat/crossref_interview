import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import requests
import numpy as np
import json
from datetime import datetime
from typing import List
import logging
from urllib.parse import quote
import time
import helper as helper

import plotly.express as px

import os
from dotenv import load_dotenv

load_dotenv()
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")
HERE = Path(__file__).parent
DATA = HERE / "data"
PARALLEL_PROCESSING = False
OUTPUT = HERE.parent / "output"

# Get the current month
current_month = datetime.now().strftime("%Y-%m")

# Define the cache file paths
ror_cache_file_path = DATA / f"ror_cache_{current_month}.json"
marple_cache_file_path = DATA / f"marple_cache_{current_month}.json"


# Load the ROR cache from the JSON file if it exists
if ror_cache_file_path.exists():
    with ror_cache_file_path.open("r") as ror_cache_file:
        ror_cache = json.load(ror_cache_file)
else:
    ror_cache = {}

# Load the Marple cache from the JSON file if it exists
if marple_cache_file_path.exists():
    with marple_cache_file_path.open("r") as marple_cache_file:
        marple_cache = json.load(marple_cache_file)
else:
    marple_cache = {}


def main():
    crossref_df = pd.read_csv(
        DATA.joinpath("crossref_ror_ids_with_affiliation_strings.csv")
    )
    ror_registry_df = pd.read_csv(DATA.joinpath("ror_registry_slim.csv"))
    ror_registry_dict = ror_registry_df.set_index("id").to_dict(orient="index")
    wikidata_ror_dataset = pd.read_csv(DATA.joinpath("wikidata_ror_dataset.csv"))
    wikidata_ror_dict = wikidata_ror_dataset.set_index("ROR_ID").to_dict(orient="index")

    tqdm.pandas()
    crossref_df["match_type"] = crossref_df.progress_apply(
        detect_lexical_match,
        axis=1,
        ror_registry_dict=ror_registry_dict,
        wikidata_ror_dict=wikidata_ror_dict,
    )

    update_match_type_with_deepseek(crossref_df, ror_registry_dict)

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

    # Convert the list of unique names to a "|" separated string
    unique_names["unique_names"] = unique_names["unique_names"].apply(
        lambda x: " | ".join(x)
    )

    # Merge the result back to no_match_counts
    no_match_counts = no_match_counts.merge(unique_names, on="ROR_ID")
    unique_prefixes = (
        no_match_df["DOI"]
        .apply(lambda x: x.split("/")[0])
        .groupby(no_match_df["ROR_ID"])
        .unique()
        .reset_index()
    )
    unique_prefixes.columns = ["ROR_ID", "unique_prefixes"]

    # Now, for each prefix, calculate the number of envolved unique DOIs.
    prefixes_df = no_match_df.assign(
        prefix=no_match_df["DOI"].apply(lambda x: x.split("/")[0])
    )
    prefix_counts = (
        prefixes_df.groupby(["ROR_ID", "prefix"])["DOI"]
        .nunique()
        .reset_index()
        .groupby("ROR_ID")
        .apply(lambda x: dict(zip(x["prefix"], x["DOI"])))
        .reset_index()
        .rename(columns={0: "prefix_counts"})
    )

    # Finally, for these prefixes we will query the crossref api and get the member name-> ["message"]["name"]
    # The information will also be cached
    # The result will be a dictionary with the prefix as key and the member name as value

    prefix_to_name = {}

    for prefix in tqdm(unique_prefixes["unique_prefixes"].explode().unique()):
        if prefix not in prefix_to_name:
            prefix_to_name[prefix] = get_member_name_from_prefix(prefix)
    # Now add a column for member counts
    prefix_counts["member_counts"] = prefix_counts["prefix_counts"].apply(
        lambda x: {prefix: prefix_to_name.get(prefix, "Not found") for prefix in x}
    )

    no_match_counts = no_match_counts.merge(prefix_counts, on="ROR_ID").drop(
        columns="id"
    )
    no_match_count_summary_html = no_match_counts.to_html(index=False)

    # Calculate the total DOIs involved for each prefix
    prefix_doi_counts = (
        prefixes_df.groupby("prefix")["DOI"]
        .nunique()
        .reset_index()
        .rename(columns={"DOI": "total_dois"})
    )

    # Calculate the total different strings involved for each prefix
    prefix_string_counts = (
        prefixes_df.groupby("prefix")["normalized_name"]
        .nunique()
        .reset_index()
        .rename(columns={"normalized_name": "total_strings"})
    )

    # Merge the two tables
    problematic_prefixes = prefix_doi_counts.merge(prefix_string_counts, on="prefix")

    # Sort by total DOIs and total strings
    problematic_prefixes = problematic_prefixes.sort_values(
        ["total_dois", "total_strings"], ascending=False
    )

    # Add names
    problematic_prefixes["member_names"] = problematic_prefixes["prefix"].apply(
        lambda x: prefix_to_name.get(x, "Not found")
    )

    # For each prefix, calculate the proportion of DOIs in the original table
    #  and the "no-match" table that it represents

    # Calculate the total DOIs in the original table
    all_dois = crossref_df["DOI"]
    all_unmatched_dois = no_match_df["DOI"]
    total_dois = all_dois.nunique()
    total_unmatched_dois = all_unmatched_dois.nunique()

    # Calculate the proportion of DOIs in the original table
    problematic_prefixes["proportion_original"] = problematic_prefixes["prefix"].apply(
        lambda x: all_dois.str.startswith(x).sum() / total_dois
    )

    # Calculate the proportion of DOIs in the "no-match" table
    problematic_prefixes["proportion_no_match"] = problematic_prefixes["prefix"].apply(
        lambda x: all_unmatched_dois.str.startswith(x).sum() / total_unmatched_dois
    )

    # Calculate the proportion growth from the original table to the "no-match" table
    problematic_prefixes["proportion_ratio"] = (
        problematic_prefixes["proportion_no_match"]
        / problematic_prefixes["proportion_original"]
    )

    # Convert to HTML
    problematic_prefixes_html = problematic_prefixes.to_html(index=False)

    # Create an HTML file showing the decision tree for the matching process
    decision_tree_html = """
    <h2>Decision Pipeline for ROR Matching</h2>
    <p>Matching is done in the following order (normalizing names):</p>
    <ol>
        <li>Try and match the provided name to the ROR display name, labels, aliases and acronyms (aka "ROR strings").</li>
        <li>If not, check if one of ROR strings is a substring of the provided name (comma-separated, space separated or just substring). <li>
        <li>If not, check if provided name is a substring of the ROR strings </li>
        <li>If not, check if the provided name is a substring of Wikidata labels/aliases</li>
        <li>If not, check if the provided name is retrieved by the ROR affiliation matching API (any position)</li>
        <li>If not, check if the provided name-ROR pair matches the manual medical school mappings</li>
        <li>If not, check if the ROR status is "withdrawn"</li>
        <li>If not, check if the provided name is retrieved by the Marple service (single-search)</li>
        <li>If not, check if DeepSeekV3 considers it a match</li>

           </ol>
    """

    # Now create a summarized match table that aggregates the counts for the groups in the decision pipeline
    # It should group the different groups in the categories described in the decision tree
    # Use the names in the selection function as a guide

    # Apply the mapping to match_summary_clean
    match_summary_clean = crossref_df["match_type"].value_counts().reset_index()
    match_summary_clean.columns = ["match_type", "count"]
    match_summary_clean["category"] = match_summary_clean["match_type"].apply(
        map_match_type_to_category
    )

    # Group by category and sum the counts
    match_summary_clean = (
        match_summary_clean.groupby("category")["count"].sum().reset_index()
    )

    # Sort by count
    match_summary_clean = match_summary_clean.sort_values("count", ascending=False)
    match_summary_clean_html = match_summary_clean.to_html(index=False)

    # Apply the mapping to crossref_df and save
    crossref_df["match_category"] = crossref_df["match_type"].apply(
        map_match_type_to_category
    )

    crossref_df.to_csv(
        DATA.joinpath("crossref_ror_ids_with_affiliation_strings_and_matches.tsv"),
        sep="\t",
        index=False,
    )

    # Define the HTML template with DataTables.js and jQuery
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ROR Matching Analysis</title>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.min.css">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.min.js"></script>
        <script>
            $(document).ready(function() {{
                $('table').DataTable({{
                    "pageLength": 5
                }});
            }});
        </script>
    </head>
    <body>
        {content}
    </body>
    </html>
    """

    # Generate the HTML content combining the above plots and tables
    html_parts = []
    html_parts.append(decision_tree_html)

    html_parts.append("<h2>Match Summary (grouped)</h2>")
    html_parts.append("<p>Note: greedy matching algorithm</p>")
    html_parts.append(match_summary_clean_html)

    # Now a pizza plot for the different groups
    fig = px.pie(
        match_summary_clean,
        values="count",
        names="category",
        title="Match Summary (grouped)",
    )

    figure_html = fig.to_html(full_html=False)
    html_parts.append(figure_html)

    html_parts.append("<h2>Match Summary (complete)</h2>")
    html_parts.append("<p>Note: greedy matching algorithm</p>")

    html_parts.append(match_summary_html)
    html_parts.append("<h2>No Match Counts</h2>")
    html_parts.append(no_match_count_summary_html)

    # Add the missing plots
    html_parts.append("<h2>Problematic Prefixes</h2>")
    html_parts.append(problematic_prefixes_html)

    html_content = "\n".join(html_parts)
    # Combine the HTML template with the content
    html_output = html_template.format(content=html_content)

    # Write the HTML output to a file
    with open(OUTPUT.joinpath("ror_matching_analysis.html"), "w") as f:
        f.write(html_output)


def update_match_type_with_deepseek(crossref_df, ror_registry_dict):
    entries_without_a_match_for_processing_with_deepseek = get_entries_witout_a_match(
        crossref_df, ror_registry_dict
    )

    llm_matches_cache_file_path = DATA / f"llm_cache_{current_month}.json"

    # Load the LLM cache from the JSON file if it exists
    if llm_matches_cache_file_path.exists():
        with llm_matches_cache_file_path.open("r") as llm_cache_file:
            llm_match_list = json.load(llm_cache_file)
    else:
        llm_match_list = []

    tuples_in_llm_match_list = [
        (entry["display"], entry["name"]) for entry in llm_match_list
    ]

    entries_to_process = []
    for entry in entries_without_a_match_for_processing_with_deepseek:
        affiliation_name = entry["name"]
        ror_name = entry["display"]
        if (ror_name, affiliation_name) not in tuples_in_llm_match_list:
            entries_to_process.append(entry)
    if len(entries_to_process) > 0:
        chunks = len(entries_to_process) / 30
        chunks = np.ceil(chunks).astype(int)
        entries_to_process_chunks = np.array_split(entries_to_process, chunks)

        for test_dataset_chunk in tqdm(entries_to_process_chunks):
            # Get the completions for the current chunk
            llm_matches = guess_matches_for_dataset(test_dataset_chunk)
            llm_matches = json.loads(llm_matches)

            try:
                llm_match_list_for_this_chunk = llm_matches["test_dataset"]
                for llm_match in llm_match_list_for_this_chunk:
                    if (
                        llm_match["display"],
                        llm_match["name"],
                    ) not in tuples_in_llm_match_list:
                        llm_match_list.append(llm_match)

                # Save the updated cache
                with llm_matches_cache_file_path.open("w") as llm_cache_file:
                    json.dump(llm_match_list, llm_cache_file, indent=4, sort_keys=True)

            except Exception as e:
                print(f"Error in LLM completion: {e}, continuing")

    # Convert llm_match_list to a dictionary for faster lookups
    llm_match_dict = {
        (entry["name"], entry["display"]): entry["match"] for entry in llm_match_list
    }

    # Update match_type for the crossref_df based on the LLM guesses
    for i, row in tqdm(crossref_df.iterrows(), total=crossref_df.shape[0]):
        affiliation_name = row["Affiliation_Name"]
        ror_display = ror_registry_dict.get(row["ROR_ID"], {}).get(
            "names.types.ror_display", ""
        )
        key = (row["Affiliation_Name"], ror_display)
        if key in llm_match_dict and llm_match_dict[key] == True:
            crossref_df.at[i, "match_type"] = "deepseek-v3 match"

    return crossref_df


def get_entries_witout_a_match(crossref_df, ror_registry_dict):
    entries_without_a_match_for_processing_with_deepseek = []

    for i, row in crossref_df.iterrows():
        if row["match_type"] == "no match":
            ror_display_name = ror_registry_dict.get(row["ROR_ID"], {}).get(
                "names.types.ror_display", ""
            )
            entries_without_a_match_for_processing_with_deepseek.append(
                {"name": row["Affiliation_Name"], "display": ror_display_name}
            )

    # Ensure unique pairs
    entries_without_a_match_for_processing_with_deepseek = [
        dict(t)
        for t in {
            tuple(d.items())
            for d in entries_without_a_match_for_processing_with_deepseek
        }
    ]

    return entries_without_a_match_for_processing_with_deepseek


def detect_lexical_match(row, ror_registry_dict, wikidata_ror_dict):
    """
    A greedy algorithm to detect if an affiliation string and a ROR ID match.

    """
    ror_id = row["ROR_ID"]
    affiliation_name = row["normalized_name"]

    if ror_id not in ror_registry_dict:
        return "ror not registered"

    ror_data = ror_registry_dict[ror_id]
    ror_id_acronyms = extract_clean_labels(ror_data["names.types.acronym.normalized"])
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
        inside_the_parenthesis = affiliation_name.split("(")[1].split(")")[0].strip()
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
        if any(ror_acronym in affiliation_name for ror_acronym in ror_id_acronyms):
            match_type = "acronym as substring"  # Note: may add noise; some acronyms may be commons strings

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
        non_normalized_affiliation_name = row["Affiliation_Name"]
        ror_candidates = get_ror_api_candidates_for_affiliation(
            affiliation_name=non_normalized_affiliation_name, local=False
        )
        ror_candidate_ids = [candidate["id"] for candidate in ror_candidates]
        ror_candidate_match_types = [
            candidate["matching_type"] for candidate in ror_candidates
        ]
        if ror_id in ror_candidate_ids:
            ror_id_position_among_candidates = ror_candidate_ids.index(ror_id)
            api_match_type = ror_candidate_match_types[ror_id_position_among_candidates]
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
                medical_school in affiliation_name for medical_school in medical_schools
            ):
                match_type = "medical school to university mapping (substring)"

    if match_type == "no match" and ror_id == "https://ror.org/04qw24q55":
        match_type = "no match (special case, Wageningen University)"

    if match_type == "no match" and ror_id == "https://ror.org/00yq55g44":
        match_type = "no match (special case, Witten/Herdecke University)"  # Systematic issue envolving multiple prefixes

    if match_type == "no match" and ror_id == "https://ror.org/03pvyf116":
        match_type = "no match (special case, Dana-Farber/Harvard Cancer Center)"  # Weird issue, similar to Witten/Herdecke. I suspect the slash is causing issues.

    if match_type == "no match":
        non_normalized_affiliation_name = row["Affiliation_Name"]

        marple_candidates = get_marple_service_candidates_for_affiliation(
            non_normalized_affiliation_name
        )
        marple_candidate_ids = [candidate["id"] for candidate in marple_candidates]
        if ror_id in marple_candidate_ids:
            marple_id_position_among_candidates = marple_candidate_ids.index(ror_id)
            match_type = "marple single-search match position " + str(
                marple_id_position_among_candidates + 1
            )

    return match_type


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


def guess_matches_for_dataset(test_dataset):
    system_prompt = """
You need to check if the affiliation corresponds to the provided ROR display name. 
It is okay if the match is, say, a parent organization.

You need to complete the dataset in JSON, adding "match: true or "match:" false for each entry.

Return a JSON, no explanation needed.

The dataset should look like this:
INPUT: 

test_dataset = [
    {
        "name": "university of north carolina at chapel hill",
        "display": "University of Notre Dame",
    },
    {
        "name": "university of north carolina at chapel hill",
        "display": "University of North Carolina at Chapel Hill",
    },
    {
        "name": "woods institute for the environment, stanford university",
        "display": "Stratford University",
    },
]

OUTPUT: 
{"test_dataset" : [
    {
        "name": "university of north carolina at chapel hill",
        "display": "University of Notre Dame",
        "match": false
    },
    {
        "name": "university of north carolina at chapel hill",
        "display": "University of North Carolina at Chapel Hill",
        "match": true
    },
    {
        "name": "woods institute for the environment, stanford university",
        "display": "Stratford University",
        "match": false
    },
]
}
"""
    client = OpenAI(api_key=DEEP_SEEK_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": str(test_dataset),
            },
        ],
        response_format={"type": "json_object"},
        temperature=1.0,
        stream=False,
    )
    return response.choices[0].message.content


def save_ror_cache():
    with ror_cache_file_path.open("w") as ror_cache_file:
        json.dump(ror_cache, ror_cache_file, indent=4, sort_keys=True)


def save_marple_cache(strategy="single-search"):
    with marple_cache_file_path.open("w") as marple_cache_file:
        json.dump(marple_cache, marple_cache_file, indent=4, sort_keys=True)


def get_marple_service_candidates_for_affiliation(affiliation_name: str) -> List[str]:
    # Check if the result is in the cache
    if affiliation_name in marple_cache:
        return marple_cache[affiliation_name]

    # URL-encode the affiliation name
    encoded_affiliation_name = quote(affiliation_name)

    # Query the Marple API
    url = f"https://marple.research.crossref.org/match?task=affiliation-matching&input={encoded_affiliation_name}&strategy=affiliation-single-search"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        # Extract relevant information from the API response
        result_marple_ids = [
            {
                "id": match["id"],
                "confidence": match["confidence"],
            }
            for match in data["message"]["items"]
        ]

    else:
        logging.error(
            f"Failed to fetch data from Marple API for {affiliation_name}: {response.status_code}"
        )
        result_marple_ids = []

    # Store the result in the cache
    marple_cache[affiliation_name] = result_marple_ids
    save_marple_cache()
    time.sleep(0.5)  # Politeness helps
    return result_marple_ids


def get_ror_api_candidates_for_affiliation(
    affiliation_name: str, local=False
) -> List[str]:
    # Check if the result is in the cache
    if affiliation_name in ror_cache:
        return ror_cache[affiliation_name]

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
    ror_cache[affiliation_name] = result_ror_ids
    save_ror_cache()
    time.sleep(0.5)
    return result_ror_ids


# Function to map match_type to category
def map_match_type_to_category(match_type):
    # Define the category mapping
    category_mapping = {
        "ror_display": "ror_display",
        "label": "label",
        "alias": "alias",
        "acronym": "acronym",
        "wikidata": "Wikidata match",
        "ror affiliation endpoint": "ROR API match",
        "medical school to university mapping": "medical school mapping",
        "withdrawn": "withdrawn ROR",
        "marple": "Marple match",
    }
    for key, value in category_mapping.items():
        if key in match_type:
            return value
    return match_type


def get_member_name_from_prefix(prefix):
    prefix_cache_file_path = DATA / f"prefix_cache_{current_month}.json"
    if prefix_cache_file_path.exists():
        with prefix_cache_file_path.open("r") as f:
            prefix_cache = json.load(f)
    else:
        prefix_cache = {}

    if prefix in prefix_cache:
        return prefix_cache[prefix]

    url = f"https://api.crossref.org/prefixes/{prefix}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            member_name = data["message"]["name"]
        except KeyError:
            member_name = "Not found"
    else:
        member_name = "Not found"

    prefix_cache[prefix] = member_name
    with prefix_cache_file_path.open("w") as f:
        json.dump(prefix_cache, f, indent=4, sort_keys=True)
    return member_name


if __name__ == "__main__":
    main()
