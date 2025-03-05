import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from pathlib import Path


import helper as helper

HERE = Path(__file__).parent
DATA = HERE / "data"
PARALLEL_PROCESSING = False


def main():
    # -------------------------------
    # Part 1: Exploratory Analysis of Crossref Data
    # -------------------------------

    input_file = DATA / "crossref_affiliation_ids.csv"
    crossref_df = pd.read_csv(input_file)
    crossref_df = crossref_df[crossref_df["ROR_ID"].str.contains("ror.org", na=False)]

    # Save intermediate file 01
    crossref_df.to_csv(DATA.joinpath("crossref_ror_ids.csv"), index=False)

    unique_rors = crossref_df["ROR_ID"].dropna().unique()
    total_unique_rors = len(unique_rors)
    crossref_df["DOI_Prefix"] = crossref_df["DOI"].str.split("/").str[0]

    group_member = (
        crossref_df.groupby("DOI_Prefix")
        .agg(
            total_entries=("DOI", "count"),
            unique_dois=("DOI", "nunique"),
            unique_rors=("ROR_ID", "nunique"),
        )
        .reset_index()
    )

    print(group_member.head())

    doi_ror_counts = crossref_df.groupby("DOI")["ROR_ID"].count()
    mean_rors_per_doi = doi_ror_counts.mean()
    median_rors_per_doi = doi_ror_counts.median()

    # Additional Metrics & Custom Binning
    ror_counts = crossref_df["ROR_ID"].value_counts().reset_index()
    ror_counts.columns = ["ROR_ID", "count"]

    ror_counts["usage_bin"] = ror_counts["count"].apply(bin_ror_usage)
    total_usage_bin = ror_counts.groupby("usage_bin")["count"].sum().reset_index()

    top15_rors = ror_counts.head(15)

    fig_top15_rors = px.bar(
        top15_rors,
        x="ROR_ID",
        y="count",
        title="Top 15 Most Used RORs",
        labels={"ROR_ID": "ROR ID", "count": "Usage Count"},
    )

    fig_ror_usage_bins, fig_ror_usage_bins_unique = generate_figs_for_ror_usage(
        ror_counts, total_usage_bin
    )

    fig_doi_ror_bins, fig_box_rors_per_doi = generate_figs_for_ror_per_doi(
        doi_ror_counts
    )

    unique_dois = crossref_df["DOI"].dropna().unique()
    total_unique_dois = len(unique_dois)

    # Now let's just focus on entrief with affiliation strings

    no_affiliation_string = crossref_df["Affiliation_Name"].apply(
        lambda x: pd.isna(x) or str(x).strip() == ""
    )
    proportion_missing_affiliation_string = no_affiliation_string.sum() / len(
        crossref_df
    )
    proportion_containing_affiliation_string = 1 - (
        no_affiliation_string.sum() / len(crossref_df)
    )

    # Save intermediate file 02 containing only files with affiliation strings
    crossref_df = crossref_df[~no_affiliation_string]
    crossref_df["normalized_name"] = crossref_df["Affiliation_Name"].apply(
        helper.normalize_spaces
    )
    crossref_df["normalized_name"] = crossref_df["normalized_name"].apply(
        helper.normalize_string
    )
    crossref_df.to_csv(
        DATA.joinpath("crossref_ror_ids_with_affiliation_strings.csv"), index=False
    )

    valid_affiliations = (
        crossref_df["Affiliation_Name"]
        .dropna()
        .map(helper.normalize_spaces)
        .replace("", np.nan)
        .dropna()
    )
    unique_affiliations = valid_affiliations.unique()
    total_unique_affiliations = len(unique_affiliations)

    affiliation_counts = valid_affiliations.value_counts().reset_index()
    affiliation_counts.columns = ["Affiliation_Name", "count"]
    top10_affiliations = affiliation_counts.head(10)

    fig_top10_affiliations = px.bar(
        top10_affiliations,
        x="Affiliation_Name",
        y="count",
        title="Top 10 Most Common Affiliation Names",
        labels={"Affiliation_Name": "Affiliation Name", "count": "Count"},
    )

    fig_affiliation_hist = px.histogram(
        affiliation_counts,
        x="count",
        nbins=30,
        title="Distribution of Affiliation Name Usage Frequency",
        labels={"count": "Usage Count"},
    )

    # Additional Plots
    affil_presence = crossref_df["Affiliation_Name"].apply(
        lambda x: "Missing" if pd.isna(x) or str(x).strip() == "" else "Present"
    )
    affil_presence_counts = affil_presence.value_counts().reset_index()
    affil_presence_counts.columns = ["Affiliation_Presence", "count"]

    fig_affil_pie = px.pie(
        affil_presence_counts,
        names="Affiliation_Presence",
        values="count",
        title="Proportion of Entries with vs. without Affiliation Names",
    )

    fig_bubble = px.scatter(
        group_member,
        x="total_entries",
        y="unique_rors",
        size="total_entries",
        color="DOI_Prefix",
        hover_name="DOI_Prefix",
        title="Crossref Members: Total Entries vs Unique RORs",
        labels={"total_entries": "Total Entries", "unique_rors": "Unique RORs"},
    )

    ror_names = crossref_df.groupby("ROR_ID")["normalized_name"].unique().reset_index()
    ror_names["num_names"] = ror_names["normalized_name"].apply(len)

    fig_ror_names_distribution = px.histogram(
        ror_names,
        x="num_names",
        nbins=20,
        title="Distribution of Different Affiliation Names per ROR",
        labels={
            "num_names": "Number of Different Affiliation Names",
            "count": "Frequency",
        },
    )

    top10_ror_diff_names = ror_names.sort_values(by="num_names", ascending=False).head(
        10
    )
    top10_table_html = top10_ror_diff_names.to_html(
        index=False,
        columns=["ROR_ID", "num_names", "normalized_name"],
        header=["ROR ID", "Count of Different Names", "List of Names"],
        escape=False,
    )

    # Analysis of Affiliation Names Associated with Multiple RORs
    affiliation_ror_counts = (
        crossref_df.groupby("normalized_name")
        .agg(unique_rors=("ROR_ID", "nunique"), total_occurrences=("ROR_ID", "count"))
        .reset_index()
    )

    ambiguous_affiliations = affiliation_ror_counts[
        affiliation_ror_counts["unique_rors"] > 1
    ]
    top10_ambiguous_affiliations = ambiguous_affiliations.sort_values(
        by="unique_rors", ascending=False
    ).head(10)

    fig_top10_ambiguous = px.bar(
        top10_ambiguous_affiliations,
        x="normalized_name",
        y="unique_rors",
        title="Top 10 Affiliation Names Associated with Multiple RORs",
        labels={"normalized_name": "Affiliation Name", "unique_rors": "Unique RORs"},
    )

    fig_affiliation_scatter = px.scatter(
        ambiguous_affiliations,
        x="total_occurrences",
        y="unique_rors",
        hover_data=["normalized_name"],
        title="Affiliation Occurrences vs. Unique RORs (Ambiguous Affiliations)",
        labels={"total_occurrences": "Total Occurrences", "unique_rors": "Unique RORs"},
    )

    fig_hist_unique_rors = px.histogram(
        affiliation_ror_counts,
        x="unique_rors",
        nbins=30,
        title="Distribution of Unique RORs per Affiliation Name",
        labels={"unique_rors": "Number of Unique RORs"},
    )

    top10_ambiguous_table_html = top10_ambiguous_affiliations.to_html(
        index=False,
        columns=["normalized_name", "unique_rors", "total_occurrences"],
        header=["Affiliation Name", "Unique RORs", "Total Occurrences"],
        escape=False,
    )

    # Additional Analysis: Affiliation Name Length Distribution
    crossref_df["affiliation_length"] = crossref_df["normalized_name"].str.len()
    fig_affiliation_length = px.histogram(
        crossref_df,
        x="affiliation_length",
        nbins=30,
        title="Distribution of Affiliation Name Lengths",
        labels={"affiliation_length": "Affiliation Name Length (characters)"},
    )

    # Distribution of Unique Affiliation Names per DOI
    affiliations_per_doi = crossref_df.groupby("DOI")["Affiliation_Name"].apply(
        lambda x: len(set(map(helper.normalize_spaces, x.dropna())))
    )
    fig_affiliations_per_doi = px.histogram(
        affiliations_per_doi,
        nbins=20,
        title="Distribution of Unique Affiliation Names per DOI",
        labels={"value": "Unique Affiliation Names per DOI", "count": "Number of DOIs"},
    )

    doi_df = pd.DataFrame({"DOI": list(unique_dois)[:20]})
    doi_df["DOI_Clickable"] = doi_df["DOI"]
    doi_table_html = doi_df[["DOI_Clickable"]].to_html(
        index=False, header=["DOI"], escape=False
    )

    # Crossref Member Details (Top 10 Members)
    group_member_top = (
        group_member.sort_values(by="total_entries", ascending=False).head(5).copy()
    )
    group_member_top["DOI_Prefix"] = group_member_top["DOI_Prefix"].apply(
        lambda m: f'<a href="https://api.crossref.org/members?query={m}" target="_blank">{m}</a>'
    )
    member_table_html = group_member_top.to_html(
        index=False,
        columns=["DOI_Prefix", "total_entries", "unique_rors"],
        header=["DOI Prefix", "Total Entries", "Unique RORs"],
        escape=False,
    )

    # -------------------------------
    # N. ROR Official Name Matching and Data Quality Analysis
    # -------------------------------

    # Load the ROR registry data from your provided CSV file.
    ror_registry_file = (
        DATA / "v1.59-2025-01-23-ror-data/v1.59-2025-01-23-ror-data_schema_v2.csv"
    )
    ror_registry_df = pd.read_csv(ror_registry_file)
    print(ror_registry_df.head())

    # Make a slimmer version of the ROR registry

    # Keep fields for id, for names, for related organizations and for Wikidata ids
    ror_registry_df = ror_registry_df[
        [
            "id",
            "names.types.acronym",
            "names.types.alias",
            "names.types.label",
            "names.types.ror_display",
            "relationships",
            "external_ids.type.wikidata.all",
        ]
    ]
    print(len(ror_registry_df))

    # -------------------------------
    # Compose the Final HTML Report
    # -------------------------------
    summary_html = f"""
    <h2>Exploratory Data Insights</h2>
    <ul>
      <li><strong>Total unique ROR IDs:</strong> {total_unique_rors}</li>
      <li><strong>Total unique DOIs:</strong> {total_unique_dois}</li>
      <li><strong>Total unique Affiliation Names:</strong> {total_unique_affiliations}</li>
      <li><strong>Proportion of entries with a ROR but no Affiliation Name:</strong> {proportion_missing_affiliation_string:.2%}</li>
      <li><strong>Mean number of RORs per DOI:</strong> {mean_rors_per_doi:.2f}</li>
      <li><strong>Median number of RORs per DOI:</strong> {median_rors_per_doi:.2f}</li>
    </ul>
    """

    fig_total = px.bar(
        group_member,
        x="DOI_Prefix",
        y="total_entries",
        title="Total Entries per Crossref Member",
        labels={"total_entries": "Total Entries", "DOI_Prefix": "Crossref Member"},
    )

    fig_unique = px.bar(
        group_member,
        x="DOI_Prefix",
        y="unique_rors",
        title="Unique RORs per Crossref Member",
        labels={"unique_rors": "Unique RORs", "DOI_Prefix": "Crossref Member"},
    )

    fig_top_members = px.bar(
        group_member.sort_values(by="total_entries", ascending=False).head(10),
        x="DOI_Prefix",
        y="total_entries",
        title="Top 10 Crossref Members by Total Entries",
        labels={"total_entries": "Total Entries", "DOI_Prefix": "Crossref Member"},
    )

    html_parts = []
    html_parts.append(
        "<html><head><title>Affiliation Data Analysis</title></head><body>"
    )
    html_parts.append("<h1>Affiliation Data Analysis Report</h1>")
    html_parts.append(summary_html)

    # Append original Crossref Member plots
    html_parts.append(pio.to_html(fig_total, full_html=False, include_plotlyjs="cdn"))
    html_parts.append(pio.to_html(fig_unique, full_html=False, include_plotlyjs=False))
    html_parts.append(
        pio.to_html(fig_top_members, full_html=False, include_plotlyjs=False)
    )

    # Append new ROR usage plots
    html_parts.append("<h2>Most Used RORs</h2>")
    html_parts.append(
        pio.to_html(fig_top15_rors, full_html=False, include_plotlyjs=False)
    )
    html_parts.append("<h3>Total ROR Occurrences by Usage Bin</h3>")
    html_parts.append(
        pio.to_html(fig_ror_usage_bins, full_html=False, include_plotlyjs=False)
    )
    html_parts.append("<h3>Unique ROR Count by Usage Bin</h3>")
    html_parts.append(
        pio.to_html(fig_ror_usage_bins_unique, full_html=False, include_plotlyjs=False)
    )

    # Append custom binning for RORs per DOI
    html_parts.append("<h2>RORs per DOI (Custom Bins)</h2>")
    html_parts.append(
        pio.to_html(fig_doi_ror_bins, full_html=False, include_plotlyjs=False)
    )

    # Append Affiliation Name metrics
    html_parts.append("<h2>Affiliation Name Metrics</h2>")
    html_parts.append(
        pio.to_html(fig_top10_affiliations, full_html=False, include_plotlyjs=False)
    )
    html_parts.append(
        pio.to_html(fig_affiliation_hist, full_html=False, include_plotlyjs=False)
    )

    # Append additional insights
    html_parts.append("<h2>Additional Insights</h2>")
    html_parts.append(
        pio.to_html(fig_affil_pie, full_html=False, include_plotlyjs=False)
    )
    html_parts.append(
        pio.to_html(fig_box_rors_per_doi, full_html=False, include_plotlyjs=False)
    )
    html_parts.append(pio.to_html(fig_bubble, full_html=False, include_plotlyjs=False))

    # Append analysis of different names per ROR
    html_parts.append("<h2>Different Affiliation Names per ROR</h2>")
    html_parts.append(
        pio.to_html(fig_ror_names_distribution, full_html=False, include_plotlyjs=False)
    )
    html_parts.append("<h3>Top 10 RORs with the Most Different Affiliation Names</h3>")
    html_parts.append(top10_table_html)

    # Append analysis: Affiliation Names Associated with Multiple RORs
    html_parts.append("<h2>Affiliation Names Associated with Multiple RORs</h2>")
    html_parts.append("<h3>Top 10 Ambiguous Affiliation Names</h3>")
    html_parts.append(
        pio.to_html(fig_top10_ambiguous, full_html=False, include_plotlyjs=False)
    )
    html_parts.append(
        "<h3>Scatter Plot: Occurrences vs. Unique RORs for Ambiguous Affiliations</h3>"
    )
    html_parts.append(
        pio.to_html(fig_affiliation_scatter, full_html=False, include_plotlyjs=False)
    )
    html_parts.append("<h3>Distribution of Unique RORs per Affiliation Name</h3>")
    html_parts.append(
        pio.to_html(fig_hist_unique_rors, full_html=False, include_plotlyjs=False)
    )
    html_parts.append("<h3>Top 10 Ambiguous Affiliation Names Table</h3>")
    html_parts.append(top10_ambiguous_table_html)

    # Append additional analysis: Affiliation Name Length Distribution
    html_parts.append("<h2>Affiliation Name Length Distribution</h2>")
    html_parts.append(
        pio.to_html(fig_affiliation_length, full_html=False, include_plotlyjs=False)
    )

    # Append new analysis: Distribution of Unique Affiliation Names per DOI
    html_parts.append("<h2>Distribution of Unique Affiliation Names per DOI</h2>")
    html_parts.append(
        pio.to_html(fig_affiliations_per_doi, full_html=False, include_plotlyjs=False)
    )

    # Append new section: Clickable DOIs Table (Top 20)
    html_parts.append("<h2>Clickable DOIs (Top 20)</h2>")
    html_parts.append(doi_table_html)

    # Append new section: Crossref Member Details with API Query Results
    html_parts.append("<h2>Crossref Member Details (Top 10 by Total Entries)</h2>")
    html_parts.append(member_table_html)

    # -------------------------------
    # Append new Section N: ROR Official Name Matching and Data Quality Analysis
    # -------------------------------

    html_parts.append("</body></html>")
    report_html = "\n".join(html_parts)

    output_file = HERE / "analysis_report.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_html)

    print(f"HTML report generated: {output_file}")


def generate_figs_for_ror_per_doi(doi_ror_counts):
    doi_ror_bins = doi_ror_counts.apply(bin_rors_per_doi).value_counts().reset_index()
    doi_ror_bins.columns = ["RORs_per_DOI_bin", "count"]

    bin_order = [str(i) for i in range(1, 10)] + ["10-50", "50+"]
    doi_ror_bins["RORs_per_DOI_bin"] = pd.Categorical(
        doi_ror_bins["RORs_per_DOI_bin"], categories=bin_order, ordered=True
    )
    doi_ror_bins = doi_ror_bins.sort_values("RORs_per_DOI_bin")

    fig_doi_ror_bins = px.bar(
        doi_ror_bins,
        x="RORs_per_DOI_bin",
        y="count",
        title="Distribution of RORs per DOI (Custom Bins)",
        labels={
            "RORs_per_DOI_bin": "Number of RORs per DOI",
            "count": "Number of DOIs",
        },
    )

    fig_box_rors_per_doi = px.box(
        x=doi_ror_counts,
        points="all",
        title="Box Plot of RORs per DOI",
        labels={"x": "Number of RORs per DOI"},
    )

    return fig_doi_ror_bins, fig_box_rors_per_doi


def generate_figs_for_ror_usage(ror_counts, total_usage_bin):
    unique_ror_bin_counts = (
        ror_counts.groupby("usage_bin").size().reset_index(name="unique_ror_count")
    )
    usage_bin_order = ["1", "2-5", "5-10", "10-100", "101-1k", "1k-5k", "5k+"]

    total_usage_bin["usage_bin"] = pd.Categorical(
        total_usage_bin["usage_bin"], categories=usage_bin_order, ordered=True
    )
    total_usage_bin = total_usage_bin.sort_values("usage_bin")

    unique_ror_bin_counts["usage_bin"] = pd.Categorical(
        unique_ror_bin_counts["usage_bin"], categories=usage_bin_order, ordered=True
    )
    unique_ror_bin_counts = unique_ror_bin_counts.sort_values("usage_bin")

    fig_ror_usage_bins = px.bar(
        total_usage_bin,
        x="usage_bin",
        y="count",
        title="Distribution of Total ROR Occurrences by Usage Bin",
        labels={"usage_bin": "Usage Bin", "count": "Total Occurrences"},
    )

    fig_ror_usage_bins_unique = px.bar(
        unique_ror_bin_counts,
        x="usage_bin",
        y="unique_ror_count",
        title="Distribution of Unique ROR Counts by Usage Bin",
        labels={"usage_bin": "Usage Bin", "unique_ror_count": "Unique ROR Count"},
    )

    return fig_ror_usage_bins, fig_ror_usage_bins_unique


def bin_ror_usage(x):
    if x == 1:
        return "1"
    elif 2 <= x <= 5:
        return "2-5"
    elif 6 <= x <= 10:
        return "5-10"
    elif 11 <= x <= 100:
        return "10-100"
    elif 101 <= x <= 999:
        return "101-1k"
    elif 1000 <= x <= 5000:
        return "1k-5k"
    else:
        return "5k+"


def bin_rors_per_doi(x):
    if x <= 9:
        return str(x)
    elif 10 <= x <= 50:
        return "10-50"
    else:
        return "50+"


if __name__ == "__main__":
    main()
