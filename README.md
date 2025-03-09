# CrossRef Data Exploration - Data Scientist interview


This repository contains the code and analysis for Problem 2, put forth by Dominika Tkaczyk as one of the tasks in process of application for a Data Scientist position at CrossRef. 

This README contains a few thoughts on the process, written before the analysis started, and kept here as a log. 

As of 2025-10-08, this analysis uses the datasets:

- April 2024 Public Data File from CrossRef (http://dx.doi.org/10.13003/849J5WP), downloaded via torrent to `src/data/crossref_raw_data`. 

- ROR Release v1.59 (https://github.com/ror-community/ror-records/releases/tag/v1.59), downloaded manually and added to `src/data/v1.59-2025-01-23-ror-data`.

- Wikidata, queried via QLever (https://qlever.cs.uni-freiburg.de/wikidata), Full Wikidata dump from https://dumps.wikimedia.org/wikidatawiki/entities (latest-all.ttl.bz2 and latest-lexemes.ttl.bz2, version 29.01.2025)

The analysis are done mostly in Python with some supporting JS for viz and exploration.

The HTML are available as a GitHub Pages - powered static website at https://tiago.bio.br/crossref_interview . 

An overview of how to run the pipeline can be seen at [./src/pipeline.sh](./src/pipeline.sh).

The pipeline was tailored to run on a simple machine (512GB SSD, 16GB of RAM, 12 intel i5 cores) on Ubuntu 24.04.1 LTS, using Python 3.12.3. 

The general view of the pipeline was drawn using Lucidchart (https://lucid.app/lucidchart/3c243585-649b-4c8b-8da4-f49ae3f54cd0/view).   

## The problem
 (by Dominika Tkaczyk)

Context: Consider this example metadata record in the Crossref API. It represents a research output. Note the following metadata fields:

    “DOI” - the DOI identifier of the output, in this case 10.14232/iskkult.2023.8.19
    “author” - information about the authors of the research output
    “author/affiliation” - information about organisations the authors are affiliated with
    “author/affiliation/name” - the name of the affiliated organisation, it can also be a full affiliation string (see this example)
    “author/affiliation/id” - identifiers of the affiliated organisation; if “id-type” is “ROR”, it means that the identifier is the ROR ID

ROR IDs are persistent identifiers assigned by the Research Organisation Registry. You can access the records of ROR organisations through the ROR REST API, here is an example.

Problem: The names and affiliation strings, as well as ROR IDs in the Crossref database are provided by Crossref members. Currently, we do not verify whether the names are consistent with the ROR database. We may have cases where we have a particular ROR ID, but the corresponding name is not the actual name of this organisation. We would like to check whether the names and affiliation strings in the Crossref database match the official organisation names from the ROR Registry.

Data sources: Over 160M records in our database. This data can be accessed through the Crossref API. Monthly data dumps are also available. ROR registry is available through a separate ROR REST API, and also as data dumps.

# Leading questions 

**What data would you need to answer the questions and how can it be gathered using available data sources?**

ROR mappings of ids to names and aliases; API available, dump very manageable ([https://zenodo.org/records/14728473 34MB csv]).

For each crossref work, a set of ROR IDs and names. For a prototype, the API. For batch, generate an intermediate dataset from the CrossRef dump. 

**What kind of tools or techniques would you use to answer the questions? How would you communicate the findings?**

I like the idea of (1) intermediate datasets and (2) interactive tools when exploring data. 

First I'd generate a pipeline to extract just the bits of the data I am interested in at the moment. 

Then, create a little app (Flask, for starters) where I could play with, say, submitting a DOI and comparing the affiliation strings with ROR. 

The quest to  *verify whether the names are consistent with the ROR database* has (at least) 2 facets: 

* Checking if names are _exactly_ the same, so as to improve aliases in ROR
* Checking if names _correspond to the same entities_

The first case is easier, where a simple string match with some fuzzyness to allow for typos seems okay. 

The second one is slightly harder, as entities may be represented in different ways, or even different languages. 

It is important to use tools that the Crossref team uses, so we can talk the same language.

Similar to ["reference parsing"](https://www.crossref.org/blog/reference-matching-for-real-this-time/), the legacy method, we could parse the affiliations into a structured format and compare the structures. 

We could use the name string in the Crossref entry to search ROR and see if the top match is the provided ID, e.g. modifying the Search-Based Matching with Validation (SBMV) method. Or we could extract named entities, using say, spaCy, and compare these entities.  Or use some Small Language Model locally and do it "brute-forcey", asking the oracle if they are both the same. Or even a larger model, if we are feeling rich. 

Wikidata has a [reasonable coverage of ROR IDs](https://www.wikidata.org/wiki/Property:P6782) too, and could play a role in the middle as a multilingual broker of labels/aliases. 

Of course, any of these would need some testing before applying to the full dataset. 

After getting an idea of how many ids+names we have in Crossref, we could create a small dataset (say, 1000 examples, handcurated or synthetic) and test which of the methods allow us to detect names that look off more effectively. 

We could use F1 scores for starters, F0.5  if the goal is batch applying to data, F2 if there is a human curation layer. 

Communication of the findings would depend on the findings and on the audience. 

For the tech team, findings may be the different scores for the methods on the test dataset, and follow classic data science viz, say, bar plots showing Precision, Recall, F1, F0.5 and F2 scores

For members / wider audience, it is interesting to show the percentage of mismatches overall, facet by different contexts (year, members) and try and find some trends. 

Saving the mistakes and trying to classify them (are they language mistakes? Are they mixing parent and child organizations?) could be interesting too. 

For ROR, it may make sense to provide a stream of matches that are linguistically different, but semantically the same (it could be a source of alias). Providing that metric is also relevant. 

**What kind of tools would you use for a one-off analysis? What if we wanted to monitor the situation over time?**

For starters, I'd like a simple app that I can enter a DOI and have it compare affiliations with ROR. 

With that, I get experience to build an analysis for the whole dataset, giving a bird-eye view of the situation. 

For monitoring over time, getting a feed of new DOI deposits, running through the system and detecting possible issues could be interesting. This could lead to a dashboard, allowing early detection of possible errors.  


** What do you see as the biggest challenge and what are some limitations?**
   
Before going through the data it may be hard to see the challenges. I think that there may be a lot of weird and different ways these ID-name pairs are provided. 

Doing it for the full Crossref dataset would need some parsing of the full, 160M record dataset to search for/extract the id-name pairs. The task could be parallelized though, as the points are independent. 

It should be very manageable, specially considering the expertise of Crossref team. 

Detecting the right and wrong pairs will come always with tradeoffs. Do we care more about getting those clearly wrong matches, or the similar, harder ones? As I see, deciding on the details of the question is never simple and requires clear communication. 

Names in ROR might be weird or wrong too, though that is likely to be rare. Again, would need some data exploration. 
 
   
**Are there any interesting additional questions or paths we might want to explore?**

I like this problem because it opens the gates to many relevant questions. Tracking affiliation is notoriously hard, and [this is likely to be relevant in practice in the near future](https://ror.org/blog/2025-01-09-metadata-matching-beyond-correctness/). 

Maybe members provide affiliations in different ways, but consistently. Faceting by member could be interesting. 

Maybe members in non-English speaking countries, or authors from non-English speaking countries, deposit affiliations with non-English names. This could be an interesting statistic.


## Good examples found in the preliminary analysis

10.14232/actahisp.2023.28.9-26
10.14232/ejqtde.2024.1.73
10.14232/iskkult.2024.10.67
10.14232/kapocs.2022.2.85-95
10.14232/sua.2024.57.4
10.14232/mped.2024.2.111 --> most interesting case
10.14232/jengeo-2024-45530