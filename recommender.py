import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from datetime import datetime

# Load dataset
df = pd.read_csv("FAR-Trans-Data/transactions.csv")

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter to 'Buy' transactions only
df = df[df['transactionType'] == 'Buy'].copy()

# Choose weighting method: 'units' or 'totalValue'
weight_column = 'units'  # or 'totalValue'

# Calculate time decay factor
decay_lambda = 0.005
latest_time = df['timestamp'].max()
df['days_since'] = (latest_time - df['timestamp']).dt.days
df['time_decay'] = np.exp(-decay_lambda * df['days_since'])

# Final weight = interaction strength * time decay
df['weight'] = df[weight_column] * df['time_decay']

# Build user-item matrix with weights
user_item_matrix = df.pivot_table(index='customerID', columns='ISIN', values='weight', aggfunc='sum', fill_value=0)

# Convert to sparse matrix
sparse_matrix = csr_matrix(user_item_matrix.values)

# Compute item-item cosine similarity
item_similarity = cosine_similarity(sparse_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Recommendation function
def recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_vector = user_item_matrix.loc[user_id]
    interacted_items = user_vector[user_vector > 0].index.tolist()
    
    scores = pd.Series(dtype=float)
    for item in interacted_items:
        item_score = item_similarity_df[item] * user_vector[item]  # weight by interaction strength
        scores = scores.add(item_score, fill_value=0)

    scores = scores.drop(labels=interacted_items, errors='ignore')  # remove already seen
    top_recommendations = scores.sort_values(ascending=False).head(top_n)

    return top_recommendations.index.tolist()

# Example usage
user_id = '00017496858921195E5A'
recommendations = recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=5)

print(f"Top 5 recommended ISINs for user {user_id}: {recommendations}")
