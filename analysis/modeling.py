# Import pandas and load the Excel file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering and text processing libraries
# Reference: https://huggingface.co/sentence-transformers
# Load BERT model from Hugging Face
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap.umap_ as umap

# Model Implementation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score

def dataloader(original_file_path="ResearchOutputs.xlsx", enriched_file_path="ResearchOutputs Group6.xlsx"):
    # Load the original data from the Excel file
    df_original = pd.read_excel(original_file_path, sheet_name=0)

    # Preview the first 5 rows to verify it loaded correctly
    print("First 5 rows of the original dataset:")
    print(df_original.head(5))

    # Load the enriched data from Step1
    df_enriched = pd.read_excel(enriched_file_path)

    # Preview the first 5 rows to verify it loaded correctly
    print("First 5 rows of the enriched dataset:")
    print(df_enriched.head(5))

    # Merge the datasets by concatenating the rows
    df = pd.concat([df_original, df_enriched], ignore_index=True)

    return df

"""# Data Exploration and Wrangling"""

def data_explore_wrangling(df):
    # Get number of rows and columns
    print(f"Data shape: {df.shape}")

    # Count of missing (NaN) values in each column
    print(f"Count of missing (NaN) values in each column: {df.isna().sum()}")

    # Through exploration of the combined dataset, 
    # we find that there are a lot of missing values within some of the columns which will greatly impact the model prediction, 
    # and also some columns are independent w.r.t other columns which is hard to give useful information in the dataset, 
    # so we only select the useful columns and form the sub-dataset."""

    # Select useful columns to form a sub-dataset
    selected_columns = [
        'ProjectStatus',
        'ProjectTitle',
        'ProjectRDC',
        'ProjectStartYear',
        'ProjectEndYear',
        'OutputTitle',
        'OutputType',
        'OutputYear',
        'OutputStatus',
        'OutputVenue'
    ]

    # Create the subdataset with only selected columns
    df_sub = df[selected_columns].copy()

    # Cap the ProjectEndYear at 2025 and handle missing values
    df_sub['ProjectDuration'] = np.where(
        df_sub['ProjectStartYear'].notnull(),
        np.where(
            df_sub['ProjectEndYear'].notnull(),
            np.minimum(df_sub['ProjectEndYear'], 2025) - df_sub['ProjectStartYear'],
            2025 - df_sub['ProjectStartYear']
        ),
        np.nan  # if ProjectStartYear is missing, result is none
    )

    # Check the value counts of the potentail target variable
    venue_counts = df_sub['OutputType'].value_counts()
    print(f"The value counts of 'OutputType': {venue_counts}")

    # Check range of ProjectDuration
    print("Range of ProjectDuration")
    print("Min:", df_sub['ProjectDuration'].min())
    print("Max:", df_sub['ProjectDuration'].max())
    return df_sub

""" 
Based on our data exploration, we aim to address two main questions.
First, can we predict the OutputType, which is the publication type of the paper, by using information such as OutputTitle, ProjectRDC, Project Duration, and ProjectStatus?
Second, We can estimate their Project Duration by using information such as ProjectTitle and ProjectRDC?
"""
"""
Prediction on OutputType
Predict the OutputType, which is the publication OutputType of the paper, by using information such as OutputTitle, ProjectRDC, ProjectDuration, and ProjectStatus?
"""

## Further Data Processing

def data_process_insight1(df_sub):

    # Drop rows where OutputTitle is missing
    df_cleaned = df_sub.dropna(subset=['OutputTitle'])

    # Check for duplicate rows in OutputTitle
    print("\nDuplicated rows before dropping:")
    print(df_cleaned.duplicated(subset=['OutputTitle']).sum())

    # Drop duplicates
    df_cleaned = df_cleaned.drop_duplicates(subset=['OutputTitle'])

    # Show cleaned dataset size and preview
    print(f"\nRemaining rows after cleaning: {df_cleaned.shape[0]}")

    return df_cleaned

def text_processing(df_cleaned, text_column):
    # Use a pretrained BERT model from the SentenceTransformers library
    # model = SentenceTransformer("microsoft/deberta-v3-small")
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    # Encode the OutputTitle column into vector embeddings
    titles = df_cleaned[text_column].tolist()
    embeddings = model.encode(titles, show_progress_bar=True)

    return embeddings

"""Clustering"""

def update_clustering_column_insight1(df_cleaned, n_clusters=5):

    # Use n_clusters = 5 to try to fit titles into 5 groups
    embeddings = text_processing(df_cleaned, 'OutputTitle')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Add the cluster results back to the dataframe
    df_cleaned['ClusterID'] = cluster_labels

    # Preview titles by cluster
    print("Added 'ClusterID' columns by clustering similar 'OutputTitle'")

    # Reduce embeddings to 2D for visualization
    mapreducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_2d = mapreducer.fit_transform(embeddings)

    # Plot each cluster separately with label
    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(cluster_labels):
        idx = cluster_labels == cluster_id
        plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], label=f'Cluster {cluster_id}', alpha=0.7)

    plt.title("Projection of BERT Embeddings for OutputTitle")
    plt.xlabel("First dimension")
    plt.ylabel("Second dimension")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_cleaned

"""Feature Engineering"""

def feature_engineering_insight1(df_cleaned):

    # Select feature columns and label
    feature_columns = ['ProjectStatus', 'ProjectRDC', 'ProjectDuration', 'OutputVenue', 'ClusterID']

    target_column = 'OutputType'
    df_model = df_cleaned[feature_columns + [target_column]].copy()

    # Filter target categories with more than 10 occurrences
    value_counts = df_model[target_column].value_counts()

    # Keep only labels with >10 occurrences
    valid_labels = value_counts[value_counts > 10].index

    # Filter the dataset
    df_model = df_model[df_model[target_column].isin(valid_labels)].copy()

    print(f"Remaining classes in {target_column}: {df_model[target_column].nunique()}")
    print(f"Remaining dataset rows: {df_model.shape[0]}")

    df_model = df_model.dropna(subset=feature_columns + [target_column])

    # One-hot encode categorical columns
    categorical_cols = ['ProjectStatus', 'ProjectRDC', 'OutputVenue', 'ClusterID']
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols)

    # Separate X and y
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)

    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder

"""PCA"""

def pca_insight1(X_train, X_test):
    # Apply PCA to reduce dimensionality to keep 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Original dimension: {X_train.shape[1]}, Reduced dimension: {X_train_pca.shape[1]}")

    # Plot the PCA result
    plt.figure(figsize=(8,5))
    num_components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(num_components, cumulative_variance, marker='o', label='Cumulative Explained Variance')

    # Horizontal line at 95% threshold
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Threshold')

    # Labels and formatting
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Variance Explained by Components')
    plt.legend()
    plt.grid(True)
    plt.show()

    return X_train_pca, X_test_pca

"""Logestic Regression"""

def logistic_regression_insight1(X_train_pca, X_test_pca, y_train_encoded, y_test_encoded, label_encoder):
    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_pca, y_train_encoded)

    # Predict on test set
    y_pred = lr.predict(X_test_pca)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)

    # Plot heatmap using seaborn
    plt.figure(figsize=(8, 6))
    labels = label_encoder.classes_
    sns.heatmap(cm, annot=True, cmap='Reds', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix by Using Logistic Regression")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.tight_layout()
    plt.show()

def random_forest_insight1(X_train_pca, X_test_pca, y_train_encoded, y_test_encoded, label_encoder):
    
    # Initialize and train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_pca, y_train_encoded)

    # Predict on the test set
    y_pred_rf = rf.predict(X_test_pca)

    # Accuracy
    accuracy_rf = accuracy_score(y_test_encoded, y_pred_rf)
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred_rf)

    # Plot heatmap using seaborn
    plt.figure(figsize=(8, 6))
    labels = label_encoder.classes_
    sns.heatmap(cm, annot=True, cmap='Reds', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix by Using Logistic Regression")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.tight_layout()
    plt.show()

"""
Prediction on Project Duration
We can estimate their Project Duration by using information such as ProjectTitle and ProjectRDC?
"""

"""Further Data Processing"""

# Drop rows where ProjectTitle is missing
def data_process_insight2(df_sub):
    df_cleaned2 = df_sub.dropna(subset=['ProjectTitle'])

    # Check for duplicate rows in ProjectTitle
    print("\nDuplicated rows before dropping:")
    print(df_cleaned2.duplicated(subset=['ProjectTitle']).sum())

    # Drop duplicates
    df_cleaned2 = df_cleaned2.drop_duplicates(subset=['ProjectTitle'])

    # Show cleaned dataset size and preview
    print(f"\nRemaining rows after cleaning: {df_cleaned2.shape[0]}")

    return df_cleaned2

"""Clustering"""

def update_clustering_column_insight2(df_cleaned2, n_clusters=5):

    # Use n_clusters = 5 to try to fit titles into 5 groups
    embeddings = text_processing(df_cleaned2, 'ProjectTitle')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Add the cluster results back to the dataframe
    df_cleaned2['Cluster_ID'] = cluster_labels

    # Plot the 2D graph distribution of the point based on the ProjectTitle
    # Reduce embeddings to 2D for visualization
    mapreducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_2d = mapreducer.fit_transform(embeddings)

    # Plot each cluster separately with label
    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(cluster_labels):
        idx = cluster_labels == cluster_id
        plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], label=f'Cluster {cluster_id}', alpha=0.7)

    plt.title("Projection of BERT Embeddings for ProjectTitle")
    plt.xlabel("First dimension")
    plt.ylabel("Second dimension")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_cleaned2

"""Feature Engineering"""

def feature_engineering_insight2(df_cleaned2):
    # Select feature columns and label
    feature_columns = ['ProjectStatus', 'ProjectRDC', 'OutputType', 'Cluster_ID']

    target_column = 'ProjectDuration'
    df_model = df_cleaned2[feature_columns + [target_column]].copy()
    df_model.head(5)

    df_model = df_model.dropna(subset=feature_columns + [target_column])
    df_model.shape

    # One-hot encode categorical columns
    categorical_cols = feature_columns
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols)

    # Separate X and y
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

"""PCA"""

def pca_insight2(X_train, X_test):

    # Apply PCA to reduce dimensionality to keep 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Original dimension: {X_train.shape[1]}, Reduced dimension: {X_train_pca.shape[1]}")

    # Plot the PCA result
    plt.figure(figsize=(8,5))
    num_components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(num_components, cumulative_variance, marker='o', label='Cumulative Explained Variance')

    # Horizontal line at 95% threshold
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Threshold')

    # Labels and formatting
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Variance Explained by Components')
    plt.legend()
    plt.grid(True)
    plt.show()

    return X_train_pca, X_test_pca

"""Linear Regression"""

def linear_regression_insight2(X_train_pca, X_test_pca, y_train, y_test):
    # Train linear regression
    lr = LinearRegression()
    lr.fit(X_train_pca, y_train)

    # Predict on test set
    y_pred = lr.predict(X_test_pca)

    # Evaluate with MSE and R²
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression Mean Squared Error (MSE): {mse:.4f}")
    print(f"Linear Regression R² Score: {r2:.4f}")

def insight1(df_sub):
    """
    Insight 1: Predict the OutputType, which is the publication OutputType of the paper,
         by using information such as OutputTitle, ProjectRDC, ProjectDuration, and ProjectStatus

    Steps:
    1. Preprocess the dataset: clean missing values, filter categories
    2. Apply semantic clustering on 'OutputTitle' using BERT embeddings
    3. Perform one-hot encoding, label encoding, and prepare feature/target sets
    4. Apply PCA to reduce dimensionality
    5. Train and evaluate both Logistic Regression and Random Forest classifiers
    """
    print("Insight1: Predict the OutputType")
    df_clean = data_process_insight1(df_sub)
    df_clean = update_clustering_column_insight1(df_clean, n_clusters=5)
    X_train, X_test, y_train_encoded, y_test_encoded, label_encoder = feature_engineering_insight1(df_clean)
    X_train_pca, X_test_pca = pca_insight1(X_train, X_test)
    logistic_regression_insight1(X_train_pca, X_test_pca, y_train_encoded, y_test_encoded, label_encoder)
    random_forest_insight1(X_train_pca, X_test_pca, y_train_encoded, y_test_encoded, label_encoder)


def insight2(df_sub):
    """
    Insight 2: Predict ProjectDuration for completed projects based on ProjectTitle and ProjectRDC

    Steps:
    1. Preprocess the dataset: filter completed projects and ensure valid duration labels
    2. Apply semantic clustering on 'ProjectTitle' using BERT embeddings
    3. Perform one-hot encoding, label encoding, and prepare feature/target sets
    4. Apply PCA for dimensionality reduction
    5. Train and evaluate a Linear Regression model to estimate ProjectDuration
    """
    print("Insight1: Predict the ProjectDuration")
    df_clean2 = data_process_insight2(df_sub)
    df_clean2 = update_clustering_column_insight2(df_clean2, n_clusters=5)
    X_train, X_test, y_train, y_test = feature_engineering_insight2(df_clean2)
    X_train_pca, X_test_pca = pca_insight2(X_train, X_test)
    linear_regression_insight2(X_train_pca, X_test_pca, y_train, y_test)


def general_process(original_file_path="ResearchOutputs.xlsx", enriched_file_path="ResearchOutputs Group6.xlsx"):
    # Full pipeline function that orchestrates data loading, preprocessing, and both insights analysis
    df = dataloader(original_file_path, enriched_file_path)
    df_sub = data_explore_wrangling(df)

    # Perform both insight
    insight1(df_sub)
    insight2(df_sub)

if __name__ == "__main__":
    general_process()