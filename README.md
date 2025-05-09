# CIT 5900 Project 3: FSRDC Research Output Analysis

This project explores research outputs associated with the Federal Statistical Research Data Centers (FSRDCs).  
We consolidated scattered data sources, enriched metadata using APIs, and prepared the data for analysis and modeling.

---

## Project Structure

```
.
├── analysis/
│   ├── merge_clean.py         # Merges and cleans raw CSV files
│   ├── deduplicate_match.py   # Deduplicates data and matches metadata
│   └── enrich_data.py         # Enriches records using OpenAlex / CrossRef APIs
│
├── raw/                       # Raw input CSVs (8 files)
├── data/                      # Supplementary metadata (ProjectsAllMetadata.xlsx)
├── output/                    # All intermediate & final results
├── main.py                    # Main script that runs all three steps in sequence
├── README.md                  # Project overview (this file)
└── requirements.txt           # Python dependencies
```
---

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

## 🧪 Completed Modules

###  Step 1: Merging & Cleaning
- Combine 8 CSVs
- Remove records with missing DOI/URL/type
- Standardize column formats

### Step 2: Deduplication & Matching
- Deduplicate by DOI or title
- Match co-authors and metadata using `ProjectsAllMetadata.xlsx`

### Step 3: Data Enrichment
- Query OpenAlex and CrossRef APIs
- Enrich fields like author list, year, volume, venue
- Standardize final columns

> Output: `enriched_data.csv` (stored in `output/`)

---

## 📈 Next Steps (Coming Soon)

- 📊 **EDA and Visualizations** (Top RDCs, authors, trends over time)
- 🤖 **Modeling Analysis** (PCA, clustering, text classification)
- 🌐 **GitHub Pages Dashboard**

---


## 🌐 GitHub Pages Dashboard

> Will be added after all figures and EDA are completed.

---

## 📄 Final Deliverables

- `ResearchOutputs_GroupX.xlsx`
- `Report_GroupX.pdf`
- `main.py` + supporting scripts
- GitHub Pages link (with EDA & modeling results)
