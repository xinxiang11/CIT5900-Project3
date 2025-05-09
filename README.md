# CIT 5900 Project 3: FSRDC Research Output Analysis

This project explores research outputs associated with the Federal Statistical Research Data Centers (FSRDCs).  
We consolidated scattered data sources, enriched metadata using APIs, and prepared the data for analysis and modeling.

## Folder Structure

```
CIT5900_Project3/
│
├── analysis/                  # Python modules for each step
│   ├── merge_clean.py
│   ├── deduplicate_match.py
│   ├── enrich_data.py
│   ├── eda_analysis.py
│   └── modeling.py
│
├── data/                      # Input files
│   ├── ProjectsAllMetadata.xlsx
│   └── raw/                   # Raw input from groups
│       ├── group1.csv
│       ├── group2.csv
│       └── ... (group8.csv)
│
├── output/                    # Results and figures
│   ├── enriched_output.csv
│   ├── ResearchOutputs Group6.xlsx
│   ├── eda_figures/
│   └── model_figures/
│
├── index.html                 # Final HTML report with plots
├── main.py                    # Master script to run the project
├── requirements.txt
└── README.md
```


##  How to Run the Project

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py
```

This will:

- Merge and clean group raw files
- Deduplicate entries and match publications to projects
- Use OpenAlex and CrossRef APIs to enrich records
- Perform EDA and modeling (classification + regression)
- Save results (tables + plots) to `output/`


##  Project Steps

### Step 1: Data Merging & Cleaning
- Merges all group CSVs
- Standardizes column names
- Drops rows with no DOI or URL
- Outputs: `output/merged_cleaned.csv`

### Step 2: Deduplication & Matching
- Deduplicates by DOI and Title
- Matches publications to FSRDC projects using researcher names
- Fills missing metadata
- Outputs: `output/matched_publications.csv`

### Step 3: Metadata Enrichment
- Enriches records using CrossRef and OpenAlex:
  - Adds title, venue, year, citations, reference, etc.
- Outputs: `output/enriched_output.csv`, `output/ResearchOutputs Group6.xlsx`

### Step 4: Exploratory Data Analysis (EDA)
- Analyzes publication lag, citation distribution, and output trends
- Visualizes key patterns across RDCs and authors
- Outputs saved in `output/eda_figures/*.png`

### Step 5: Modeling & Prediction
- Task 1: Predict OutputType using PCA + Logistic Regression & Random Forest
- Task 2: Predict ProjectDuration using BERT embeddings + Linear Regression
- Uses UMAP and confusion matrices to interpret results
- All plots saved in `output/model_figures/*.png`

### Step 6: Report Packaging & Publishing
- All analysis results are compiled into `index.html`
- Project is version-controlled using Git and uploaded to GitHub:
  - Repository: [https://github.com/xinxiang11/CIT5900-Project3](https://github.com/xinxiang11/CIT5900-Project3)
- You can preview the results using GitHub Pages or by opening `index.html` in a browser.

##  Final Deliverables

- `ResearchOutputs_GroupX.xlsx`
- `Report_GroupX.pdf`
- `main.py` + supporting scripts
- GitHub Pages link (with EDA & modeling results)
