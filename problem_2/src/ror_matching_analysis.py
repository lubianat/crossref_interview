import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import difflib
from itertools import combinations
from collections import defaultdict
import requests
from joblib import Parallel, delayed
from tqdm import tqdm

import helper as helper 
HERE = Path(__file__).parent
DATA = HERE / "data"
PARALLEL_PROCESSING = False
OUTPUT = HERE.parent / "output"
def main():
    crossref_df = pd.read_csv(DATA.joinpath("crossref_ror_ids_with_affiliation_strings.csv"))
    ror_registry_df = pd.read_csv(DATA.joinpath("ror_registry_slim.csv"))
    ror_registry_dict = ror_registry_df.set_index('id').to_dict(orient='index')

    wikidata_ror_dataset = pd.read_csv(DATA.joinpath("wikidata_ror_dataset.csv"))
    wikidata_ror_dict = wikidata_ror_dataset.set_index('ROR_ID').to_dict(orient='index')
    def detect_lexical_match(row, ror_registry_dict, wikidata_ror_dict):
        ror_id = row['ROR_ID']
        affiliation_name = row['normalized_name']

        if ror_id not in ror_registry_dict:
            return "ror not registered"

        ror_data = ror_registry_dict[ror_id]
        ror_id_acronyms = extract_clean_labels(ror_data['names.types.acronym.normalized'])
        ror_id_aliases = extract_clean_labels(ror_data['names.types.alias.normalized'])
        ror_id_labels = extract_clean_labels(ror_data['names.types.label.normalized'])
        ror_id_ror_display = ror_data['names.types.ror_display.normalized']

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

        else:
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
                wikidata_labels = wikidata_ror_data.get('LABEL', "").split(" | ")
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
            elif any (ror_label in affiliation_name for ror_label in ror_id_labels):
                match_type = "label as substring"
            elif any (ror_alias in affiliation_name for ror_alias in ror_id_aliases):
                match_type = "alias as substring"
            
            # TODO - Check labels in related organizations
        return match_type

    tqdm.pandas()
    crossref_df['match_type'] = crossref_df.progress_apply(detect_lexical_match, axis=1, ror_registry_dict=ror_registry_dict, wikidata_ror_dict=wikidata_ror_dict)  
   
    crossref_df.to_csv(DATA.joinpath("crossref_ror_ids_with_affiliation_strings_and_matches.tsv"), sep="\t", index=False)
    
    # Filter df by no-match and save 
    no_match_df = crossref_df[crossref_df['match_type'] == 'no match']
    no_match_df.to_csv(DATA.joinpath("crossref_ror_ids_with_affiliation_strings_and_no_matches.tsv"), sep="\t", index=False)
        # Overall match summary
    match_summary = crossref_df['match_type'].value_counts().reset_index()
    match_summary.columns = ['match_type', 'count']
    match_summary_html = match_summary.to_html(index=False)
    
    # Per ROR summary (counts and exact match proportion)
    ror_match_summary = crossref_df.groupby('ROR_ID')['match_type'].value_counts().unstack(fill_value=0).reset_index()
    ror_match_summary['total'] = ror_match_summary.select_dtypes(include=[np.number]).sum(axis=1)
    ror_match_summary['exact_prop'] = ror_match_summary.get('exact', 0) / ror_match_summary['total']
    ror_match_summary_html = ror_match_summary.to_html(index=False)
    
   
    # Second filter: Records with a partial match (to filter out cases like a department mentioning the University name)
    partial_matches = crossref_df[crossref_df['match_type'].isin(['comma-separated subpart label match', 'comma-separated subpart alias match', 'comma-separated subpart acronym match'])]
    fig_partial = px.histogram(partial_matches, x='ROR_ID',
                               title="Distribution of Partial Matches by ROR",
                               labels={'ROR_ID': 'ROR ID', 'count': 'Number of Partial Matches'})


    # Research hypothesis: are dois that have one "no-match" more likely to have a second "no-match"?
    # In other words, is there an association between individual DOIs and having a "no-match" — my guess is yes. 
   
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
    html_parts.append("<h2>ROR Match Summary</h2>")
    html_parts.append(ror_match_summary_html)
    html_parts.append("<h2>Partial Matches</h2>")
    html_parts.append(fig_partial.to_html(full_html=False, include_plotlyjs='cdn'))
    html_output = "\n".join(html_parts)
    with open(OUTPUT.joinpath("ror_matching_analysis.html"), "w") as f:
        f.write(html_output)



def extract_clean_labels(ror_id_labels):
    if pd.isna(ror_id_labels):
        return []
    clean_labels = [label.split(": ")[1] if ": " in label else label for label in ror_id_labels.split("; ")]
    clean_labels = [label.lower() for label in clean_labels]
    clean_labels = [label.strip() for label in clean_labels]
    return clean_labels

if __name__ == "__main__":
    main()