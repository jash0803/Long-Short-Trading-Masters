import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from datetime import datetime

st.set_page_config(page_title="ItemCF Recommender", layout="wide")
st.title("📈 Item-Based Collaborative Filtering for Financial Asset Recommendation")


df = pd.read_csv("FAR-Trans-Data/transactions.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filters
df = df[df['transactionType'] == 'Buy'].copy()

st.subheader("⚙️ Settings")
col1, col2 = st.columns(2)

with col1:
    weight_column = st.selectbox("Select Weighting Column", ['units', 'totalValue'])
with col2:
    decay_lambda = st.slider("Select Time Decay (λ)", min_value=0.001, max_value=0.02, step=0.001, value=0.005)

latest_time = df['timestamp'].max()
df['days_since'] = (latest_time - df['timestamp']).dt.days
df['time_decay'] = np.exp(-decay_lambda * df['days_since'])
df['weight'] = df[weight_column] * df['time_decay']

# Create user-item matrix
user_item_matrix = df.pivot_table(index='customerID', columns='ISIN', values='weight', aggfunc='sum', fill_value=0)
sparse_matrix = csr_matrix(user_item_matrix.values)

# Item-item similarity
item_similarity = cosine_similarity(sparse_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# User selection
st.subheader("👤 Recommend for a User")
selected_user = st.selectbox("Select a Customer ID", user_item_matrix.index)
top_n = st.slider("Top-N Recommendations", 1, 20, 5)

def recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_vector = user_item_matrix.loc[user_id]
    interacted_items = user_vector[user_vector > 0].index.tolist()

    scores = pd.Series(dtype=float)
    for item in interacted_items:
        item_score = item_similarity_df[item] * user_vector[item]
        scores = scores.add(item_score, fill_value=0)

    scores = scores.drop(labels=interacted_items, errors='ignore')
    top_recommendations = scores.sort_values(ascending=False).head(top_n)
    return top_recommendations

recs = recommend_items(selected_user, user_item_matrix, item_similarity_df, top_n=top_n)

st.subheader("📌 Top Recommendations")
if not recs.empty:
    st.table(recs.reset_index().rename(columns={"index": "ISIN", 0: "Score"}))
else:
    st.warning("No recommendations found for this user.")

st.markdown("---")
st.caption("Powered by Item-based Collaborative Filtering with time-decayed weighted interactions.")
