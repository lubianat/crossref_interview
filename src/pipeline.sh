# python3 parse_crossref_dump_and_extract_data.py

# Print to screen
echo "Running pipeline for Problem 2"

echo "Running parse_crossref_dump_and_extract_data.py"
python3 generate_wikidata_ror_dataset.py 

echo "Running crossref_exploration_and_pre-process.py"
python3 crossref_exploration_and_pre-process.py

echo "Running preprocess_ror_ids.py"
python3 preprocess_ror_ids.py

echo "Running ror_matching_analysis.py" 
python3 ror_matching_analysis.py