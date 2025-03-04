import pandas as pd
from pathlib import Path
from helper import normalize_string

HERE = Path(__file__).parent
DATA = HERE / "data"
OUTPUT = HERE.parent / "output"


def main():
    full_ror_df = pd.read_csv(
        DATA / "v1.59-2025-01-23-ror-data" / "v1.59-2025-01-23-ror-data_schema_v2.csv"
    )

    # SLIM example
    #                               id,names.types.acronym,names.types.alias,names.types.label,names.types.ror_display,relationships,external_ids.type.wikidata.all,names.types.acronym.normalized,names.types.alias.normalized,names.types.label.normalized,names.types.ror_display.normalized
    # https://ror.org/04ttjf776,no_lang_code: RMIT,en: Royal Melbourne Institute of Technology University,en: RMIT University,RMIT University,"child: https://ror.org/039p7nx39, https://ror.org/03m3ca021, https://ror.org/004axh929; related: https://ror.org/010mv7n52",Q1057890,no_lang_code: rmit,en: royal melbourne institute of technology university,en: rmit university,rmit university

    ror_df_slim = full_ror_df[
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
    # Add normalized
    ror_df_slim["names.types.acronym.normalized"] = ror_df_slim[
        "names.types.acronym"
    ].apply(normalize_string)
    ror_df_slim["names.types.alias.normalized"] = ror_df_slim[
        "names.types.alias"
    ].apply(normalize_string)
    ror_df_slim["names.types.label.normalized"] = ror_df_slim[
        "names.types.label"
    ].apply(normalize_string)
    ror_df_slim["names.types.ror_display.normalized"] = ror_df_slim[
        "names.types.ror_display"
    ].apply(normalize_string)

    # Save slimmed down version
    ror_df_slim.to_csv(DATA / "ror_registry_slim.csv", index=False)


if __name__ == "__main__":
    main()
