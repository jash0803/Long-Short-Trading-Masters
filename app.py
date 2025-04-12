import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

########################################
# 1. DATA LOADING & PREPROCESSING
########################################
def load_data():
    # Load the CSV files (assumes UTF-8 encoding)
    asset_df = pd.read_csv("FAR-Trans-Data/asset_information.csv")
    customer_df = pd.read_csv("FAR-Trans-Data/customer_information.csv")
    transactions_df = pd.read_csv("FAR-Trans-Data/transactions.csv")
    limit_prices_df = pd.read_csv("FAR-Trans-Data/limit_prices.csv")
    
    # Fix any column naming issues
    if 'ustomerID' in transactions_df.columns:
        transactions_df.rename(columns={'ustomerID': 'customerID'}, inplace=True)
    
    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(asset_df, customer_df, transactions_df):
    # Work only with "Buy" transactions (as a positive signal)
    trans_buy = transactions_df[transactions_df['transactionType'] == "Buy"].copy()
    
    # Aggregate transactions: you can use count or sum of totalValue if available.
    # Here we use count as a proxy for interest.
    rating_df = trans_buy.groupby(['customerID', 'ISIN']).size().reset_index(name="count")
    
    # Build the user-item rating matrix
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='count').fillna(0)
    
    return rating_matrix, rating_df

########################################
# 2. COLLABORATIVE FILTERING COMPONENT
########################################
def matrix_factorization(rating_matrix, n_components=5):
    # Perform low-rank approximation with TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(rating_matrix)
    V = svd.components_.T  # shape: (num_assets, n_components)
    
    pred_ratings = np.dot(U, V.T)
    pred_df = pd.DataFrame(pred_ratings, index=rating_matrix.index, columns=rating_matrix.columns)
    return pred_df

########################################
# 3. CONTENT-BASED FILTERING COMPONENT
########################################
def content_based_scores(customer_id, rating_df, asset_df, limit_prices_df):
    """
    Create asset feature vectors from:
      - assetCategory and assetSubCategory (one-hot encoded)
      - Profitability from limit_prices (normalized)
    For a given customer, construct a profile by averaging the features of assets they bought.
    Then compute cosine similarity between that profile and each asset feature.
    """
    # Merge asset info with profitability info from limit_prices
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory']].copy()
    
    # Merge profitability – note: not all assets may have limit price records.
    asset_features = asset_features.merge(limit_prices_df[['ISIN', 'profitability']], on='ISIN', how='left')
    asset_features['profitability'].fillna(asset_features['profitability'].median(), inplace=True)
    
    # One-hot encode assetCategory and assetSubCategory
    cat_dummies = pd.get_dummies(asset_features['assetCategory'], prefix="cat")
    sub_dummies = pd.get_dummies(asset_features['assetSubCategory'], prefix="sub")
    
    asset_feat = pd.concat([asset_features[['ISIN', 'profitability']], cat_dummies, sub_dummies], axis=1)
    asset_feat.set_index("ISIN", inplace=True)
    
    # Normalize the profitability column to [0,1]
    asset_feat['profitability_norm'] = (asset_feat['profitability'] - asset_feat['profitability'].min()) / \
                                       (asset_feat['profitability'].max() - asset_feat['profitability'].min())
    asset_feat.drop(columns=['profitability'], inplace=True)
    
    # Build the customer profile: average features of assets they bought
    cust_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN']
    
    if len(cust_assets) > 0 and any(cust_assets.isin(asset_feat.index)):
        cust_vector = asset_feat.loc[cust_assets].mean(axis=0).values.reshape(1, -1)
        sim_scores = cosine_similarity(cust_vector, asset_feat)[0]
        content_score = pd.Series(sim_scores, index=asset_feat.index)
    else:
        # No prior transaction => use neutral score (e.g., 0.5)
        content_score = pd.Series(0.5, index=asset_feat.index)
    return content_score

########################################
# 4. DEMOGRAPHIC-BASED COMPONENT
########################################
def demographic_score(customer_id, customer_df, asset_df, limit_prices_df):
    """
    A more advanced demographic matching:
      - Uses customer's riskLevel and investmentCapacity.
      - Incorporates a simplified risk-return tradeoff using asset profitability.
    The mapping below is illustrative: aggressive or premium customers might favor assets with higher
    profitability (though with higher volatility), whereas conservative customers may prefer lower profitability assets.
    """
    # Risk mapping: assign a target profitability range based on risk level.
    risk_map = {
        "Conservative": (0, 0.4),
        "Predicted_Conservative": (0, 0.4),
        "Income": (0.3, 0.6),
        "Balanced": (0.4, 0.7),
        "Aggressive": (0.6, 1.0),
        "Predicted_Income": (0.3, 0.6),
        "Predicted_Balanced": (0.4, 0.7),
        "Predicted_Aggressive": (0.6, 1.0)
    }
    
    cust_info = customer_df[customer_df['customerID'] == customer_id]
    if cust_info.empty:
        risk = "Balanced"
    else:
        risk = cust_info.iloc[-1]['riskLevel']  # assume most recent record
    
    # Default target profitability range
    target_range = risk_map.get(risk, (0.4, 0.7))
    
    # Merge asset info with profitability
    asset_demo = asset_df[['ISIN', 'assetCategory', 'assetSubCategory']].copy()
    asset_demo = asset_demo.merge(limit_prices_df[['ISIN', 'profitability']], on='ISIN', how='left')
    asset_demo['profitability'].fillna(asset_demo['profitability'].median(), inplace=True)
    
    # Score assets higher if their profitability is close to the center of target_range.
    target_center = (target_range[0] + target_range[1]) / 2
    def score_profit(prof):
        # A simple inverse distance score normalized to [0,1]
        return 1 - min(abs(prof - target_center) / (target_center), 1)
    
    asset_demo['demo_score'] = asset_demo['profitability'].apply(score_profit)
    demo_score = pd.Series(asset_demo['demo_score'].values, index=asset_demo['ISIN'])
    return demo_score

########################################
# 5. HYBRID RECOMMENDATION COMBINING THE THREE COMPONENTS
########################################
def normalize_scores(s):
    if s.max() - s.min() > 0:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return s

def hybrid_recommendation(customer_id, rating_matrix, pred_df, rating_df, asset_df, 
                          customer_df, limit_prices_df, weights, top_n=5):
    """
    Combines:
      - Collaborative filtering (CF) score from matrix factorization.
      - Content-based (CB) score from asset features (including profitability).
      - Demographic (DEMO) score based on customer's risk and asset profitability.
      
    'weights' is a tuple: (CF_weight, CB_weight, DEMO_weight)
    """
    # 1. Collaborative Filtering
    if customer_id in pred_df.index:
        cf_scores = pred_df.loc[customer_id]
    else:
        cf_scores = pd.Series(0, index=rating_matrix.columns)
    
    # 2. Content-based Scores
    content_scores = content_based_scores(customer_id, rating_df, asset_df, limit_prices_df)
    
    # 3. Demographic-based Scores
    demo_scores = demographic_score(customer_id, customer_df, asset_df, limit_prices_df)
    
    # Normalize each score component to [0,1]
    cf_norm = normalize_scores(cf_scores)
    cb_norm = normalize_scores(content_scores)
    demo_norm = normalize_scores(demo_scores)
    
    # Weighted hybrid score
    final_score = weights[0]*cf_norm + weights[1]*cb_norm + weights[2]*demo_norm
    
    # Exclude assets that the customer has already bought
    bought_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique() if not rating_df[rating_df['customerID'] == customer_id].empty else []
    final_score = final_score.drop(labels=bought_assets, errors='ignore')
    
    recommendations = final_score.sort_values(ascending=False).head(top_n)
    return recommendations

########################################
# 6. STREAMLIT FRONTEND
########################################
def main():
    st.title("FAR-Trans Asset Recommender")
    st.write("An improved hybrid recommendation system leveraging the FAR-Trans dataset, combining collaborative filtering, enriched content-based filtering, and demographic matching based on customer risk profiles and asset profitability.")
    
    # Load all required data
    st.write("Loading datasets ...")
    asset_df, customer_df, transactions_df, limit_prices_df = load_data()
    
    # Preprocess transactions data
    st.write("Preprocessing transactions ...")
    rating_matrix, rating_df = preprocess_data(asset_df, customer_df, transactions_df)
    
    # Run collaborative filtering (matrix factorization)
    st.write("Performing matrix factorization ...")
    pred_df = matrix_factorization(rating_matrix, n_components=5)
    
    # Sidebar: parameters
    st.sidebar.header("Recommendation Settings")
    
    # Customer selection: list all customers (from rating matrix) and also allow manual entry
    customer_list = list(set(rating_matrix.index.tolist()) | set(customer_df['customerID'].unique()))
    customer_id_input = st.sidebar.selectbox("Select Customer ID", sorted(customer_list))
    
    top_n = st.sidebar.number_input("Top N Recommendations", min_value=1, value=5)
    
    st.sidebar.subheader("Component Weights (CF, Content, Demographic)")
    cf_weight = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.5)
    cb_weight = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.3)
    demo_weight = st.sidebar.slider("Demographic Weight", 0.0, 1.0, 0.2)
    
    weights = (cf_weight, cb_weight, demo_weight)
    
    # Button trigger
    if st.sidebar.button("Generate Recommendations"):
        st.write(f"Generating recommendations for customer: **{customer_id_input}**")
        recs = hybrid_recommendation(customer_id_input, rating_matrix, pred_df, rating_df, asset_df, 
                                     customer_df, limit_prices_df, weights, top_n=int(top_n))
        st.write("### Top Recommendations")
        st.write(recs.to_frame("Score"))
        
        # Show asset details for recommended assets
        rec_asset_info = asset_df[asset_df['ISIN'].isin(recs.index)]
        st.write("### Asset Details")
        st.write(rec_asset_info)
    
if __name__ == '__main__':
    main()
