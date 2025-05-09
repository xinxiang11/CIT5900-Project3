import pandas as pd
import requests
import time

def enrich_data(input_df, output_filename='ResearchOutputs_Group6.xlsx', sleep_time=1):
    """
    Full enrichment function:
    - Renames columns to standard names
    - Enriches metadata using CrossRef and OpenAlex APIs
    - Fills Researcher column if empty (from CrossRef authors)
    - Adds cited_by_count from OpenAlex
    - Adds OutputStatus based on publication dates (Published, Accepted, Unknown)
    - Generates OutputBiblio (reference format)
    - Removes rows where Researcher is empty
    - Saves the final result as both CSV and Excel
    """
    # Define the column renaming map
    column_rename = {
        'outputtitle': 'OutputTitle',
        'researcher': 'Researcher',
        'url': 'URL',
        'projectid': 'ProjectID',
        'projectpi': 'ProjectPI',
        'projectrdc': 'ProjectRDC',
        'projectstatus': 'ProjectStatus',
        'outputyear': 'OutputYear',
        'doi': 'DOI',
        'projectyearstarted': 'ProjectStartYear',
        'projectyearended': 'ProjectEndYear',
        'projecttitle': 'ProjectTitle',
        'outputbiblio': 'OutputBiblio',
        'outputtype': 'OutputType',
        'outputstatus': 'OutputStatus',
        'outputvenue': 'OutputVenue',
        'outputmonth': 'OutputMonth',
        'outputvolume': 'OutputVolume',
        'outputnumber': 'OutputNumber',
        'outputpages': 'OutputPages'
    }

    # Step 1: Rename columns to standard names
    df = input_df.rename(columns=column_rename)

    # Step 2: Loop through each row to enrich
    updated_rows = []
    for idx, row in df.iterrows():
        doi = row.get('DOI')
        print(f"Processing {idx+1}/{len(df)} - DOI: {doi}")
        enriched_row = row.copy()

        if pd.notnull(doi):
            url = f"https://api.crossref.org/works/{doi}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json().get('message', {})

                    # Fill fields from CrossRef
                    authors = []
                    for author in data.get('author', []):
                        full_name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                        authors.append(full_name)

                    # OutputTitle
                    title = data.get('title', [''])[0]
                    if title:
                        if pd.isnull(enriched_row.get('OutputTitle')) or enriched_row.get('OutputTitle') == '':
                            enriched_row['OutputTitle'] = title

                    # OutputVenue
                    venue = data.get('container-title', [''])[0]
                    if venue:
                        if pd.isnull(enriched_row.get('OutputVenue')) or enriched_row.get('OutputVenue') == '':
                            enriched_row['OutputVenue'] = venue

                    # OutputYear
                    year = data.get('published-print', {}).get('date-parts', [[None]])[0][0]
                    if not year:
                        year = data.get('issued', {}).get('date-parts', [[None]])[0][0]
                    if year:
                        if pd.isnull(enriched_row.get('OutputYear')) or enriched_row.get('OutputYear') == '':
                            enriched_row['OutputYear'] = year

                    # OutputMonth
                    month = data.get('published-print', {}).get('date-parts', [[None, None]])[0][1]
                    if month:
                        if pd.isnull(enriched_row.get('OutputMonth')) or enriched_row.get('OutputMonth') == '':
                            enriched_row['OutputMonth'] = month

                    # OutputVolume
                    volume = data.get('volume', '')
                    if volume:
                        if pd.isnull(enriched_row.get('OutputVolume')) or enriched_row.get('OutputVolume') == '':
                            enriched_row['OutputVolume'] = volume

                    # OutputNumber
                    number = data.get('issue', '')
                    if number:
                        if pd.isnull(enriched_row.get('OutputNumber')) or enriched_row.get('OutputNumber') == '':
                            enriched_row['OutputNumber'] = number

                    # OutputPages
                    pages = data.get('page', '')
                    if pages:
                        if pd.isnull(enriched_row.get('OutputPages')) or enriched_row.get('OutputPages') == '':
                            enriched_row['OutputPages'] = pages

                    # Researcher (only if empty)
                    authors_string = '; '.join(authors)
                    if authors_string:
                        if pd.isnull(enriched_row.get('Researcher')) or enriched_row.get('Researcher') == '':
                            enriched_row['Researcher'] = authors_string

                    # OutputType
                    output_type = data.get('type', None)
                    if output_type:
                        if pd.isnull(enriched_row.get('OutputType')) or enriched_row.get('OutputType') == '':
                            enriched_row['OutputType'] = output_type

                    # OutputStatus logic
                    if 'published-print' in data:
                        status = 'Published'
                    elif 'published-online' in data:
                        status = 'Published'
                    elif 'accepted' in data:
                        status = 'Accepted'
                    else:
                        status = 'Unknown'

                    if pd.isnull(enriched_row.get('OutputStatus')) or enriched_row.get('OutputStatus') == '':
                        enriched_row['OutputStatus'] = status

                    # OutputBiblio: build reference format
                    biblio = ''
                    if authors_string and year and title and venue:
                        biblio = f"{authors_string}. ({year}). {title}. {venue}"
                        if volume:
                            biblio += f", {volume}"
                            if number:
                                biblio += f"({number})"
                        if pages:
                            biblio += f", {pages}"
                        biblio += '.'

                    if biblio:
                        if pd.isnull(enriched_row.get('OutputBiblio')) or enriched_row.get('OutputBiblio') == '':
                            enriched_row['OutputBiblio'] = biblio

                else:
                    print(f"[CrossRef] DOI {doi} returned status {response.status_code}")
            except Exception as e:
                print(f"[CrossRef] Failed to fetch DOI {doi}: {e}")

            # OpenAlex: cited_by_count
            openalex_url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            try:
                response = requests.get(openalex_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    cited_count = data.get('cited_by_count', None)
                    if cited_count is not None:
                        enriched_row['cited_by_count'] = cited_count
                else:
                    print(f"[OpenAlex] DOI {doi} returned status {response.status_code}")
            except Exception as e:
                print(f"[OpenAlex] Failed to fetch DOI {doi}: {e}")

        else:
            print("No DOI found; skipped.")

        updated_rows.append(enriched_row)
        time.sleep(sleep_time)

    # Combine all enriched rows into a DataFrame
    enriched_df = pd.DataFrame(updated_rows)

    # Remove rows where Researcher is empty (NaN or empty string)
    enriched_df = enriched_df[~(enriched_df['Researcher'].isnull() | (enriched_df['Researcher'] == ''))]

    # Save as CSV and Excel inside the function
    enriched_df.to_csv('enriched_output.csv', index=False)
    print("Enriched data saved to 'enriched_output.csv'")

    enriched_df.to_excel(output_filename, index=False)
    print(f"Enriched data also saved to '{output_filename}'")

    return enriched_df