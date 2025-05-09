# CIT 5900 Project 3: FSRDC Research Output Analysis

This project explores research outputs associated with the Federal Statistical Research Data Centers (FSRDCs).  
We consolidated scattered data sources, enriched metadata using APIs, and prepared the data for analysis and modeling.

---

## Project Structure
## 📁 Folder Structure

```
CIT5900_Project3/
│
├── analysis/                 # Python modules for each step
│   ├── merge_clean.py
│   ├── deduplicate_match.py
│   ├── enrich_data.py
│   ├── eda_analysis.py
│
├── data/                     # Input files
│   └── ProjectsAllMetadata.xlsx
│   └── raw                   # data from all groups
│
├── output/                   # All intermediate and final results
│   ├── merged_cleaned.csv
│   ├── matched_publications.csv
│   ├── enriched_output.csv
│   ├── ResearchOutputs_Group6.xlsx
│   ├── *.png
│
├── main.py                   # Orchestration script to run all steps
└── README.md
└── requirements.txt
```


## How to Run the Project

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python main.py
```

This will:

- Merge and clean raw data from `raw/`
- Deduplicate and match co-authors with metadata from `data/`
- Use OpenAlex and CrossRef APIs to enrich publication records

All outputs will be saved to the `output/` folder.

---
## Project Steps

### Step 1: Data Merging & Cleaning
- Merges all group CSV files
- Standardizes column names
- Drops rows with no DOI or URL
- Outputs: `output/merged_cleaned.csv`

### Step 2: Deduplication & Matching
- Deduplicates by DOI and Title
- Matches publications to FSRDC projects via PI/Researcher names
- Fills missing metadata
- Outputs: `output/matched_publications.csv`

### Step 3: Metadata Enrichment
- Uses CrossRef and OpenAlex APIs to enrich:
  - Title, venue, year, volume, pages, citation count, etc.
- Adds APA-style reference (OutputBiblio)
- Outputs: `enriched_output.csv`, `ResearchOutputs_Group6.xlsx`

### Step 4: Exploratory Data Analysis
- Computes:
  - Top RDCs by output
  - Most cited works
  - Top authors
  - Publication lags
  - Citation vs. productivity
- Saves results in `.csv` and `.png` files in `output/`

### 🔜 Step 5: (In Progress)
- Final summary and packaging for submission


## 📄 Final Deliverables

- `ResearchOutputs_GroupX.xlsx`
- `Report_GroupX.pdf`
- `main.py` + supporting scripts
- GitHub Pages link (with EDA & modeling results)
