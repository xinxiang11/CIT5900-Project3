import pandas as pd
import matplotlib.pyplot as plt

def top_rdcs_by_output(df):
    top10 = df['ProjectRDC'].value_counts().head(10).reset_index()
    top10.columns = ['RDC', 'Count']
    top10.to_csv('top10_rdcs_by_output.csv', index=False)
    print("\nTop 10 RDCs by research output:")
    print(top10.to_string(index=False))
    return top10

def plot_publication_trend(df):
    df['OutputYear'] = pd.to_numeric(df['OutputYear'], errors='coerce')
    year_counts = df.dropna(subset=['OutputYear']).groupby('OutputYear').size().sort_index()
    year_counts.plot(marker='o')
    plt.title('Annual Publication Trend')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('fig_publication_trend.png', dpi=300)
    plt.close()
    return year_counts

def top_10_named_authors(df):
    df['Researcher'] = df['Researcher'].fillna('')
    authors = df['Researcher'].str.split(';').explode().str.strip()
    author_counts = authors[authors != ''].value_counts().head(10)
    top10 = author_counts.reset_index()
    top10.columns = ['Author', 'PublicationCount']
    top10.to_csv('top10_named_authors.csv', index=False)
    author_counts.sort_values().plot(kind='barh')
    plt.title('Top 10 Most Prolific Named Authors')
    plt.tight_layout()
    plt.savefig('fig_top10_named_authors.png', dpi=300)
    plt.close()
    return top10

def citation_insights(df):
    df['cited_by_count'] = pd.to_numeric(df['cited_by_count'], errors='coerce')
    clean = df.dropna(subset=['cited_by_count'])
    stats = clean['cited_by_count'].describe()
    print("\nCitation Statistics:")
    print(stats)
    clean['cited_by_count'].plot.hist(bins=20)
    plt.title('Citation Histogram')
    plt.tight_layout()
    plt.savefig('fig_citation_histogram.png', dpi=300)
    plt.close()
    clean['cited_by_count'].plot.box()
    plt.tight_layout()
    plt.savefig('fig_citation_boxplot.png', dpi=300)
    plt.close()
    top = clean[['OutputTitle', 'cited_by_count']].sort_values('cited_by_count', ascending=False).head(10)
    top.columns = ['OutputTitle', 'CitationCount']
    top.to_csv('top10_most_cited.csv', index=False)
    return top

def institutional_productivity_vs_citations(df):
    df['cited_by_count'] = pd.to_numeric(df['cited_by_count'], errors='coerce')
    inst_output = df.groupby('ProjectRDC')['OutputTitle'].count()
    inst_cite = df.dropna(subset=['cited_by_count']).groupby('ProjectRDC')['cited_by_count'].mean()
    stats = pd.concat([inst_output, inst_cite], axis=1)
    stats.columns = ['Productivity', 'AvgCitations']
    stats.fillna(0, inplace=True)
    stats.to_csv('institutional_productivity_vs_citations.csv')
    return stats

def publication_time_analysis(df):
    df['OutputYear'] = pd.to_numeric(df['OutputYear'], errors='coerce')
    df['ProjectStartYear'] = pd.to_numeric(df['ProjectStartYear'], errors='coerce')
    df['LagYears'] = df['OutputYear'] - df['ProjectStartYear']
    valid = df.dropna(subset=['LagYears'])
    stats = valid['LagYears'].describe()
    print("\nPublication Lag Statistics:")
    print(stats)
    valid['LagYears'].plot.hist(bins=20)
    plt.title('Publication Lag Histogram')
    plt.tight_layout()
    plt.savefig('fig_publication_lag_histogram.png', dpi=300)
    plt.close()
    valid['LagYears'].plot.box()
    plt.tight_layout()
    plt.savefig('fig_publication_lag_boxplot.png', dpi=300)
    plt.close()
    return stats

def eda_analysis(csv_path):
    df = pd.read_csv(csv_path)
    results = {
        "top_rdcs": top_rdcs_by_output(df),
        "trend": plot_publication_trend(df),
        "top_authors": top_10_named_authors(df),
        "citations": citation_insights(df),
        "institutional_stats": institutional_productivity_vs_citations(df),
        "lag_stats": publication_time_analysis(df)
    }
    return results
