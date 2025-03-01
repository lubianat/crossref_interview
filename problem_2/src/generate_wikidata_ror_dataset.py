# SPARQL Query https://qlever.cs.uni-freiburg.de/wikidata

import requests
import pandas as pd
import json
import time
import SPARQLWrapper
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from tqdm import tqdm
import helper as helper
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
OUTPUT = HERE.parent / "output"

query_01 = """
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?ror_id ?labelStr WHERE {
  ?a wdt:P6782 ?ror_id .
  ?a rdfs:label ?label .  
  ?a skos:altLabel ?altlabel .
  BIND(STR(?altlabel) AS ?labelStr)
}
"""

query_02 = """
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?ror_id ?labelStr WHERE {
  ?a wdt:P6782 ?ror_id .
  ?a rdfs:label ?label .  
  ?a skos:altLabel ?altlabel .
  BIND(STR(?label) AS ?labelStr)
}
"""

def get_wikidata_ror_data(query):
    sparql = SPARQLWrapper("https://qlever.cs.uni-freiburg.de/api/wikidata")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

def process_results(results):
    data = []
    for result in results["results"]["bindings"]:
        ror_id = result["ror_id"]["value"]
        label = result["labelStr"]["value"]
        data.append((ror_id, label))
    return data

def main():
    try:
        df_01 = pd.read_csv(DATA / "wikidata_ror_altlabels.csv")
        print("Loaded wikidata_ror_altlabels.csv")
    except FileNotFoundError:
        results_01 = get_wikidata_ror_data(query_01)
        data_01 = process_results(results_01)
        df_01 = pd.DataFrame(data_01, columns=["ROR_ID", "LABEL"])
        df_01.to_csv(DATA / "wikidata_ror_altlabels.csv", index=False)
    
    try:
        df_02 = pd.read_csv(DATA / "wikidata_ror_labels.csv")
        print("Loaded wikidata_ror_labels.csv")
    except FileNotFoundError:
        results_02 = get_wikidata_ror_data(query_02)
        data_02 = process_results(results_02)
        df_02 = pd.DataFrame(data_02, columns=["ROR_ID", "LABEL"])
        df_02.to_csv(DATA / "wikidata_ror_labels.csv", index=False)

    # Merge the two dataframes and normalize the labels (helper.normalize_string)
    # and remove the duplicates

    def extend_ror_id(ror_id):
        ror_with_https = "https://ror.org/" + ror_id
        return ror_with_https
    
    df = pd.concat([df_01, df_02], ignore_index=True)
    df["LABEL"] = df["LABEL"].apply(helper.normalize_string)
    df["ROR_ID"] = df["ROR_ID"].apply(extend_ror_id)
    # Filter out rows with empty labels
    df = df[df["LABEL"] != ""]
    df.drop_duplicates(inplace=True)
    # Make sure all labels are strings
    df["LABEL"] = df["LABEL"].astype(str)
    # group by ror and concat labels separated by comma
    df = df.groupby("ROR_ID")["LABEL"].apply(lambda x: " | ".join(x)).reset_index()
    df.to_csv(DATA / "wikidata_ror_dataset.csv", index=False)    
if __name__ == "__main__":
    main()
