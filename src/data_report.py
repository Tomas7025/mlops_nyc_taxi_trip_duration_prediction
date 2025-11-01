import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
# Missing Values Report
# Duplicated Samples
# Outliers
# 
def save_correlation_image(df: pd.DataFrame, output_path: Path):
    """Compute correlation matrix and save as heatmap image"""
    corr_matrix = df.corr(method='pearson')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", cbar=True)
    plt.title(f"Correlation Matrix - {output_path.stem}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Saved correlation image: {output_path}")

def save_outlier_boxplots(df: pd.DataFrame, output_path: Path):
    """Save boxplots of all numeric columns to visualize outliers"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("⚠️ No numeric columns for boxplot visualization.")
        return

    # Create subplots (max 12 per page)
    ncols = 3
    nrows = min(4, (len(numeric_cols) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols[:nrows * ncols]):
        sns.boxplot(x=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Boxplots for Outlier Detection - {output_path.stem}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📦 Saved boxplot outlier image: {output_path}")


def missing_values_report(dataset: pd.DataFrame):
	total_rows = len(dataset)
	missing_columns = []

	for column in dataset.columns:
		missing_count = dataset[column].isna().sum()
		if missing_count > 0:
			missing_percentage = np.round((missing_count / total_rows) * 100,2)
			print(f'Column {column} has {missing_percentage}% of missing values')
			missing_columns.append(column)
		else:
			print(f'Column {column} has no missing values')
		
	return missing_columns

def duplicated_samples_report(dataset):

	return

def outlier_report(dataset):

	return


def full_report(dataset):

	print("################ Missing Value Report ################")
	missing_values_report(dataset)

	print("################ Duplicated Samples Report ################")
	duplicated_samples_report(dataset)

	print("################ Outliers Report ################")
	outlier_report(dataset)

	return