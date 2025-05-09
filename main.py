# main.py

from analysis.merge_clean import merge_clean
from analysis.deduplicate_match import process_and_match_publications
from analysis.enrich_data import enrich_data
from analysis.eda_analysis import eda_analysis
from analysis.modeling import general_process  

def main():
    # Step 1: Merge and clean raw CSV files
    print("Step 1: Merging and cleaning raw CSV files...")
    df_cleaned = merge_clean(
        data_dir="data/raw",
        csv_pattern="group*.csv",
        out_path="output/merged_cleaned.csv",
        verbose=True
    )

    # Step 2: Deduplicate and match publications
    print("Step 2: Deduplicating and matching publications...")
    df_matched = process_and_match_publications(
        csv_path="output/merged_cleaned.csv",
        excel_path="data/ProjectsAllMetadata.xlsx",
        output_csv_path="output/matched_publications.csv"
    )

    # Step 3: Enrich metadata using APIs
    print("Step 3: Enriching data using CrossRef and OpenAlex APIs...")
    df_enriched = enrich_data(
        input_df=df_matched,
        output_filename="output/ResearchOutputs_Group6.xlsx",
        sleep_time=1
    )

    # Step 4: Run EDA and generate visualizations/statistics
    print("Step 4: Running EDA and saving visualizations...")
    results = eda_analysis("enriched_output.csv")
    print("EDA summary keys:", list(results.keys()))

    # Step 5: Predict publication type and project duration
    print("Step 5: Running modeling and prediction...")
    general_process(
        original_file_path="output/matched_publications.csv",
        enriched_file_path="output/ResearchOutputs_Group6.xlsx"
    )

    print("\n✅ All steps completed successfully.")

if __name__ == "__main__":
    main()
