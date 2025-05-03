import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('FAR-Trans-Data/asset_information.csv')

# Function to get unique assetSubCategory values for each assetCategory
def get_subcategories_by_category(dataframe):
    categories = dataframe['assetCategory'].unique()
    result = {}
    
    for category in categories:
        # Filter the dataframe by category and get unique subcategories
        subcategories = dataframe[dataframe['assetCategory'] == category]['assetSubCategory'].unique()
        # Remove NaN values and sort
        subcategories = sorted([x for x in subcategories if isinstance(x, str) and pd.notna(x)])
        result[category] = subcategories
    
    return result

# Apply the function to get the categorization
categorization = get_subcategories_by_category(df)

# Display the results
print("Asset Subcategories by Category:")
print("-" * 40)

for category, subcategories in categorization.items():
    print(f"\n{category}:")
    for subcategory in subcategories:
        print(f"  - {subcategory}")

# Count how many assets are in each category and subcategory combination
print("\n\nAsset Counts by Category and Subcategory:")
print("-" * 40)
category_subcategory_counts = df.groupby(['assetCategory', 'assetSubCategory']).size().reset_index(name='count')

for category in df['assetCategory'].unique():
    print(f"\n{category}:")
    subset = category_subcategory_counts[category_subcategory_counts['assetCategory'] == category]
    for _, row in subset.iterrows():
        if isinstance(row['assetSubCategory'], str) and pd.notna(row['assetSubCategory']):
            print(f"  - {row['assetSubCategory']}: {row['count']} assets")