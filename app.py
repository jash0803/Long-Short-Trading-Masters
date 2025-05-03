import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

########################################
# 1. DATA LOADING & PREPROCESSING
########################################
def load_data():
    # Load the CSV files (assumes UTF-8 encoding)
    asset_df = pd.read_csv("FAR-Trans-Data/asset_information.csv")
    customer_df = pd.read_csv("FAR-Trans-Data/customer_information.csv")
    transactions_df = pd.read_csv("FAR-Trans-Data/transactions.csv")
    limit_prices_df = pd.read_csv("FAR-Trans-Data/limit_prices.csv")
    
    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(transactions_df):
    # Only "Buy" as positive signal
    buys = transactions_df[transactions_df.transactionType == "Buy"].copy()
    # Sort by timestamp so that .tail(1) is most recent
    buys['timestamp'] = pd.to_datetime(buys.timestamp)
    buys = buys.sort_values('timestamp')
    return buys

def leave_one_out_split(buys):
    """For each user, hold out their last-buy as test, rest as train."""
    train_list, test_list = [], []
    for uid, grp in buys.groupby('customerID'):
        if len(grp) < 2:
            # If only one transaction, use it in train and none in test
            train_list.append(grp)
        else:
            train_list.append(grp.iloc[:-1])
            test_list.append(grp.iloc[-1:])
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=buys.columns)
    return train_df, test_df

def build_rating_matrix(train_df):
    rating_df = train_df.groupby(['customerID','ISIN']).size().reset_index(name='count')
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
    Content-based filtering using:
      - Cleaned and grouped assetCategory and assetSubCategory (one-hot)
      - Profitability from limit_prices
    """

    # Step 1: Clean and prepare asset features
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory']].copy()

    # Merge MTF subcategory 'Bonds' into 'Bond'
    asset_features['assetSubCategory'] = asset_features['assetSubCategory'].replace({'Bonds': 'Bond'})

    # Merge limit_prices for profitability
    asset_features = asset_features.merge(
        limit_prices_df[['ISIN', 'profitability']], on='ISIN', how='left'
    )
    # Fill missing profitability with median
    asset_features['profitability'] = asset_features['profitability'].fillna(asset_features['profitability'].median())

    # Group rare subcategories (<10 assets) into 'Other'
    subcat_counts = asset_features['assetSubCategory'].value_counts()
    rare_subcats = subcat_counts[subcat_counts < 10].index
    asset_features['assetSubCategory'] = asset_features['assetSubCategory'].replace(rare_subcats, 'Other')

    # One-hot encode
    cat_dummies = pd.get_dummies(asset_features['assetCategory'], prefix='cat')
    subcat_dummies = pd.get_dummies(asset_features['assetSubCategory'], prefix='sub')

    # Combine features
    asset_feat = pd.concat([asset_features[['ISIN', 'profitability']], cat_dummies, subcat_dummies], axis=1)
    asset_feat.set_index("ISIN", inplace=True)

    # Step 2: Build user profile
    cust_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN']
    cust_assets = cust_assets[cust_assets.isin(asset_feat.index)]  # Ensure valid ISINs

    if len(cust_assets) > 0:
        cust_vector = asset_feat.loc[cust_assets].mean(axis=0).values.reshape(1, -1)
        sim_scores = cosine_similarity(cust_vector, asset_feat)[0]
        content_score = pd.Series(sim_scores, index=asset_feat.index)
    else:
        # Cold start: assign neutral score
        content_score = pd.Series(0.5, index=asset_feat.index)
    return content_score

########################################
# 4. DEMOGRAPHIC-BASED COMPONENT
########################################
def demographic_score(customer_id, customer_df, asset_df):
    """
    Returns a score for each asset based on how well the assetCategory aligns with the customer's
    riskLevel and investmentCapacity.
    """
    # Simplify predicted labels to their base forms
    def normalize_label(label):
        if pd.isna(label) or label == "Not_Available":
            return None
        return label.replace("Predicted_", "")
    
    # Mappings to numeric values
    risk_map = {
        "Conservative": 1, "Income": 2, "Balanced": 3, "Aggressive": 4
    }

    cap_map = {
        "CAP_LT30K": 1,
        "CAP_30K_80K": 2,
        "CAP_80K_300K": 3,
        "CAP_GT300K": 4
    }

    # Get latest record per customer
    customer_df_sorted = customer_df.sort_values("timestamp").drop_duplicates("customerID", keep="last")
    user_info = customer_df_sorted[customer_df_sorted["customerID"] == customer_id]

    if user_info.empty:
        return pd.Series(0.5, index=asset_df["ISIN"])  # fallback if no info

    risk = normalize_label(user_info["riskLevel"].values[0])
    cap = normalize_label(user_info["investmentCapacity"].values[0])

    # If values are missing, return neutral scores
    if risk not in risk_map or cap not in cap_map:
        return pd.Series(0.5, index=asset_df["ISIN"])

    user_vector = np.array([risk_map[risk], cap_map[cap]])

    # Create average demographic vector for each assetCategory
    asset_scores = []
    for cat in asset_df["assetCategory"].unique():
        assets_in_cat = asset_df[asset_df["assetCategory"] == cat]
        customers = customer_df[customer_df["customerID"].isin(assets_in_cat["ISIN"])]  # unlikely match, so skip
        # Instead of that, assume each category is associated with all customers
        demographics = customer_df.copy()
        demographics["riskLevel"] = demographics["riskLevel"].apply(normalize_label)
        demographics["investmentCapacity"] = demographics["investmentCapacity"].apply(normalize_label)
        demographics = demographics.dropna(subset=["riskLevel", "investmentCapacity"])
        demographics = demographics[
            demographics["riskLevel"].isin(risk_map) & demographics["investmentCapacity"].isin(cap_map)
        ]

        if demographics.empty:
            avg_vector = np.array([2.5, 2.5])  # neutral default
        else:
            avg_vector = np.array([
                demographics["riskLevel"].map(risk_map).mean(),
                demographics["investmentCapacity"].map(cap_map).mean()
            ])

        # Inverse distance = similarity
        sim = 1 - np.linalg.norm(user_vector - avg_vector) / np.linalg.norm([3, 3])  # normalize distance
        asset_scores.append((cat, sim))

    category_sim_map = dict(asset_scores)

    # Assign each asset a score based on its category
    scores = asset_df["assetCategory"].map(category_sim_map).fillna(0.5)
    return pd.Series(scores.values, index=asset_df["ISIN"])

########################################
# 5. HYBRID RECOMMENDATION COMBINING THE THREE COMPONENTS
########################################
def normalize_scores(s):
    if s.max() - s.min() > 0:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return s

def hybrid_recommendation(customer_id, rating_matrix, pred_df, rating_df, asset_df, 
                          customer_df, limit_prices_df, weights, top_n):
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
    demo_scores = demographic_score(customer_id, customer_df, asset_df)
    
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

#############################
# 6. EVALUATION METRICS
#############################
def compute_rmse(pred_df, test_df):
    # only for user-item pairs in test set
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i = row['customerID'], row['ISIN']
        if (u in pred_df.index) and (i in pred_df.columns):
            y_true.append(1.0)               # held-out buy = implicit rating 1
            y_pred.append(pred_df.at[u,i])
    if not y_true:
        return None
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_recall_at_n(pred_func, train_df, test_df, rating_matrix, rating_df, asset_df, customer_df, limit_prices_df, weights,pred_ratings, N):
    """For each user in test, generate top-N and check if test item in recommendations."""
    precisions, recalls = [], []
    for _, row in test_df.iterrows():
        u, test_isin = row['customerID'], row['ISIN']
        # generate recommendations for u
        recs = pred_func(u, rating_matrix, pred_ratings, rating_df, asset_df, customer_df, limit_prices_df, weights, top_n=N)
        hit = int(test_isin in recs.index)
        precisions.append(hit / N)
        recalls.append(hit / 1)  # since there's only 1 held-out item
    if not precisions:
        return None, None
    return np.mean(precisions), np.mean(recalls)

#############################
# 7. STREAMLIT APP
#############################
def main():
    st.title("FAR-Trans Asset Recommender")
    st.write("An improved hybrid recommendation system leveraging the FAR-Trans dataset, combining collaborative filtering, enriched content-based filtering, and demographic matching based on customer risk profiles and asset profitability.")
    
    # Load & preprocess
    asset_df, customer_df, transactions_df, limit_prices_df = load_data()
    buys = preprocess_data(transactions_df)
    train_df, test_df = leave_one_out_split(buys)
    rating_matrix, rating_df = build_rating_matrix(train_df)
    
    # CF
    pred_ratings = matrix_factorization(rating_matrix, n_components=5)
    
    # Sidebar controls
    st.sidebar.header("Recommendation & Eval Settings")

    customer_list = list(rating_matrix.index)
    customer_id_input = st.sidebar.selectbox("Customer ID", customer_list)

    N = st.sidebar.number_input("Top N", min_value=1, value=5)

    eval_mode = st.sidebar.checkbox("Run Evaluation Metrics")
    st.sidebar.subheader("Component Weights (CF, Content, Demographic)")
    cf_weight = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.5)
    cb_weight = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.3)
    demo_weight = st.sidebar.slider("Demographic Weight", 0.0, 1.0, 0.2)
    
    weights = (cf_weight, cb_weight, demo_weight)
    
    # Button trigger
    if st.sidebar.button("Generate Recommendations"):
        st.write(f"Generating recommendations for customer: **{customer_id_input}**")
        recs = hybrid_recommendation(customer_id_input, rating_matrix, pred_ratings, rating_df, asset_df, 
                                     customer_df, limit_prices_df, weights, top_n=int(N))
        st.write("### Top Recommendations")
        st.write(recs.to_frame("Score"))
        
        # Show asset details for recommended assets
        rec_asset_info = asset_df[asset_df['ISIN'].isin(recs.index)]
        st.write("### Asset Details")
        st.write(rec_asset_info)
    
    if eval_mode:
        st.write("### Evaluation Metrics (Leave-One-Out)")
        rmse = compute_rmse(pred_ratings, test_df)
        precision, recall = precision_recall_at_n(
            hybrid_recommendation, train_df, test_df,
            rating_matrix, rating_df, asset_df, customer_df, limit_prices_df,
            weights, pred_ratings,N
        )
        st.write(f"RMSE on held-out buys: **{rmse:.4f}**" if rmse is not None else "No RMSE computed")
        if precision is not None:
            st.write(f"Precision@{N}: **{precision:.4f}**, Recall@{N}: **{recall:.4f}**")
        else:
            st.write("No Precision/Recall computed")

if __name__ == '__main__':
    main()
