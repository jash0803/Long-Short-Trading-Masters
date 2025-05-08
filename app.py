import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from datetime import datetime

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
    Simplified content-based filtering using:
      - Asset category and subcategory
      - Sector and industry information
      - Market information
      - Profitability metrics
    """
    # Step 1: Prepare asset features
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']].copy()
    
    # Merge profitability
    asset_features = asset_features.merge(
        limit_prices_df[['ISIN', 'profitability']], 
        on='ISIN', 
        how='left'
    )
    
    # Fill missing values with medians/modes
    asset_features['profitability'] = asset_features['profitability'].fillna(asset_features['profitability'].median())
    asset_features['sector'] = asset_features['sector'].fillna('Unknown')
    asset_features['industry'] = asset_features['industry'].fillna('Unknown')
    
    # One-hot encode categorical features
    feature_cols = ['assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']
    encoded_features = pd.get_dummies(asset_features[feature_cols])
    
    # Add profitability as a feature
    encoded_features['profitability'] = asset_features['profitability']
    
    # Set ISIN as index
    encoded_features.index = asset_features['ISIN']
    
    # Step 2: Build user profile
    # Get user's assets and convert to list for proper filtering
    user_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique().tolist()
    # Filter to only include assets that exist in our features
    user_assets = [asset for asset in user_assets if asset in encoded_features.index]
    
    if len(user_assets) == 0:
        # Cold start: return neutral scores
        return pd.Series(0.5, index=encoded_features.index)
    
    # Calculate user profile as mean of their asset features
    user_profile = encoded_features.loc[user_assets].mean()
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(
        user_profile.values.reshape(1, -1),
        encoded_features.values
    )[0]
    
    # Create series with ISINs as index
    content_scores = pd.Series(similarity_scores, index=encoded_features.index)
    
    return content_scores

########################################
# 4. DEMOGRAPHIC-BASED COMPONENT
########################################
def demographic_score(customer_id, customer_df, asset_df):
    """
    Returns a score for each asset based on how well the assetCategory aligns with the customer's
    demographic profile, including risk level, investment capacity, and other factors.
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

    # Extract basic demographic info
    risk = normalize_label(user_info["riskLevel"].values[0])
    cap = normalize_label(user_info["investmentCapacity"].values[0])
    customer_type = user_info["customerType"].values[0]

    # If values are missing, return neutral scores
    if risk not in risk_map or cap not in cap_map:
        return pd.Series(0.5, index=asset_df["ISIN"])

    # Create a more comprehensive user vector
    user_vector = np.array([
        risk_map[risk],  # Risk tolerance
        cap_map[cap],    # Investment capacity
        1 if customer_type == "Premium" else 0,  # Premium customer flag
        1 if customer_type == "Professional" else 0,  # Professional flag
    ])

    # Create average demographic vector for each assetCategory
    asset_scores = []
    for cat in asset_df["assetCategory"].unique():
        assets_in_cat = asset_df[asset_df["assetCategory"] == cat]
        
        # Get all customers who have invested in this category
        demographics = customer_df.copy()
        demographics["riskLevel"] = demographics["riskLevel"].apply(normalize_label)
        demographics["investmentCapacity"] = demographics["investmentCapacity"].apply(normalize_label)
        demographics = demographics.dropna(subset=["riskLevel", "investmentCapacity"])
        demographics = demographics[
            demographics["riskLevel"].isin(risk_map) & 
            demographics["investmentCapacity"].isin(cap_map)
        ]

        if demographics.empty:
            avg_vector = np.array([2.5, 2.5, 0.5, 0.5])  # neutral default
        else:
            avg_vector = np.array([
                demographics["riskLevel"].map(risk_map).mean(),
                demographics["investmentCapacity"].map(cap_map).mean(),
                (demographics["customerType"] == "Premium").mean(),
                (demographics["customerType"] == "Professional").mean()
            ])

        # Calculate similarity using weighted Euclidean distance
        weights = np.array([0.4, 0.3, 0.2, 0.1])  # Weights for each feature
        sim = 1 - np.sqrt(np.sum(weights * (user_vector - avg_vector) ** 2)) / np.sqrt(np.sum(weights * np.array([3, 3, 1, 1]) ** 2))
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
      - Collaborative filtering (CF) score from matrix factorization
      - Content-based (CB) score from asset features
      - Demographic (DEMO) score based on customer profile
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
    """Compute RMSE only for user-item pairs in test set."""
    if test_df.empty:
        return None
        
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i = row['customerID'], row['ISIN']
        if (u in pred_df.index) and (i in pred_df.columns):
            y_true.append(1.0)  # held-out buy = implicit rating 1
            y_pred.append(pred_df.at[u,i])
    
    if not y_true:
        return None
        
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_recall_at_n(pred_func, train_df, test_df, rating_matrix, rating_df, asset_df, customer_df, limit_prices_df, weights, pred_ratings, N):
    """Compute precision and recall at N for each user in test set."""
    if test_df.empty:
        return None, None
        
    precisions, recalls = [], []
    valid_users = 0
    
    for _, row in test_df.iterrows():
        try:
            u, test_isin = row['customerID'], row['ISIN']
            
            # Skip if user has no training data
            if u not in rating_matrix.index:
                continue
                
            # Generate recommendations for u
            recs = pred_func(u, rating_matrix, pred_ratings, rating_df, asset_df, customer_df, limit_prices_df, weights, top_n=N)
            
            # Skip if no recommendations could be generated
            if recs is None or len(recs) == 0:
                continue
                
            # Check if test item is in recommendations
            hit = int(test_isin in recs.index)
            precisions.append(hit / N)
            recalls.append(hit)  # since there's only 1 held-out item
            valid_users += 1
            
        except Exception as e:
            print(f"Error processing user {u}: {str(e)}")
            continue
    
    if valid_users == 0:
        return None, None
        
    return np.mean(precisions), np.mean(recalls)

def process_questionnaire_responses(responses):
    """
    Process questionnaire responses to determine risk level and investment capacity.
    Returns a tuple of (risk_level, investment_capacity)
    """
    # Risk level determination based on key questions
    risk_questions = {
        'q16': 0.3,  # Risk appetite
        'q17': 0.3,  # Investment expectations
        'q18': 0.2,  # Focus on gains vs losses
        'q19': 0.2   # Reaction to 20% decline
    }
    
    risk_score = 0
    for q, weight in risk_questions.items():
        if q in responses:
            answer = responses[q]
            if q == 'q16':  # Risk appetite
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
            elif q == 'q17':  # Investment expectations
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
            elif q == 'q18':  # Focus on gains vs losses
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
            elif q == 'q19':  # Reaction to decline
                risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
    
    # Map risk score to risk level
    if risk_score >= 3.5:
        risk_level = "Aggressive"
    elif risk_score >= 2.5:
        risk_level = "Balanced"
    elif risk_score >= 1.5:
        risk_level = "Income"
    else:
        risk_level = "Conservative"
    
    # Investment capacity determination
    if 'q13' in responses:  # Amount of funds available to invest
        investment = responses['q13']
        if investment == 'a':
            investment_capacity = "CAP_GT300K"
        elif investment == 'b':
            investment_capacity = "CAP_80K_300K"
        elif investment == 'c':
            investment_capacity = "CAP_30K_80K"
        else:
            investment_capacity = "CAP_LT30K"
    else:
        investment_capacity = "CAP_LT30K"  # Default to lowest capacity
    
    return risk_level, investment_capacity

def update_customer_profile(customer_id, risk_level, investment_capacity, customer_df):
    """Update customer profile with new questionnaire responses"""
    new_row = pd.DataFrame({
        'customerID': [customer_id],
        'customerType': ['Mass'],  # Default type
        'riskLevel': [risk_level],
        'investmentCapacity': [investment_capacity],
        'lastQuestionnaireDate': [datetime.now().strftime('%Y-%m-%d')],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    
    # Append new row to customer_df
    updated_df = pd.concat([customer_df, new_row], ignore_index=True)
    return updated_df

def compute_roi_at_k(recommendations, limit_prices_df, k=10):
    """
    Compute Return on Investment (ROI) for top-k recommendations.
    ROI is calculated using the profitability metric from limit_prices_df.
    """
    if recommendations is None or len(recommendations) == 0:
        return None
        
    # Get top-k recommendations
    top_k = recommendations.head(k)
    
    # Get profitability for recommended assets
    roi_values = limit_prices_df.set_index('ISIN')['profitability'].loc[top_k.index]
    
    # Calculate average ROI
    avg_roi = roi_values.mean()
    
    return avg_roi

def compute_ndcg_at_k(recommendations, test_df, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at k.
    Uses the test set transactions as relevance indicators.
    """
    if recommendations is None or len(recommendations) == 0:
        return None
        
    # Get top-k recommendations
    top_k = recommendations.head(k)
    
    # Create relevance list (1 if item is in test set, 0 otherwise)
    relevance = [1 if isin in test_df['ISIN'].values else 0 for isin in top_k.index]
    
    # Calculate DCG
    dcg = 0
    for i, rel in enumerate(relevance):
        dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal case: all relevant items are at the top)
    idcg = 0
    num_relevant = sum(relevance)
    for i in range(min(num_relevant, k)):
        idcg += 1 / np.log2(i + 2)
    
    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg

#############################
# 7. STREAMLIT APP
#############################
def main():
    st.title("FAR-Trans Asset Recommender")
    st.write("An improved hybrid recommendation system leveraging the FAR-Trans dataset, combining collaborative filtering, enriched content-based filtering, and demographic matching.")
    
    # Display author information
    st.markdown("---")
    st.markdown("Created by: [Jash Shah](https://www.linkedin.com/in/jashshah0803/)")
    st.markdown("---")
    
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

    N = st.sidebar.number_input("Top N", min_value=1, max_value=20, value=10)  # Changed default to 10

    eval_mode = st.sidebar.checkbox("Run Evaluation Metrics")
    st.sidebar.subheader("Component Weights")
    
    # Initialize session state for weights
    if 'weights' not in st.session_state:
        st.session_state.weights = [0.4, 0.3, 0.3]  # Default weights
    
    # Create sliders for weights
    cf_weight = st.sidebar.slider(
        "Collaborative Filtering Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weights[0],
        step=0.1,
        key="cf_weight"
    )
    
    cb_weight = st.sidebar.slider(
        "Content-Based Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weights[1],
        step=0.1,
        key="cb_weight"
    )
    
    demo_weight = st.sidebar.slider(
        "Demographic Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weights[2],
        step=0.1,
        key="demo_weight"
    )
    
    # Update weights list with current slider values
    st.session_state.weights = [cf_weight, cb_weight, demo_weight]
    
    weights = tuple(st.session_state.weights)
    
    # Add questionnaire section
    st.header("Risk Assessment Questionnaire")
    st.write("Please answer the following questions to help us better understand your investment profile.")
    
    # Initialize session state for questionnaire responses
    if 'questionnaire_responses' not in st.session_state:
        st.session_state.questionnaire_responses = {}
    
    # Key risk assessment questions
    questions = {
        'q16': "How would you rate your appetite for 'risk'?",
        'q17': "Which of the following sentences best fits your investment expectations?",
        'q18': "In the event that you have to make a financial decision, are you more concerned with potential losses or potential gains?",
        'q19': "Assuming that the value of your investment declines by 20% in short period of time, then your risk tolerance would be:",
        'q13': "What is the amount of funds you have invested or have available to invest?",
        'q6': "How would you describe your level of investment knowledge?",
        'q7': "What is your investment experience?",
        'q8': "How often on average did you make trades in various financial instruments in the last three years?"
    }
    
    options = {
        'q16': {
            'a': "Particularly high. I really like to take risk.",
            'b': "Probably high. I usually like to take risks.",
            'c': "Moderate. I like to take the occasional risk.",
            'd': "Low. I usually don't like to take risks.",
            'e': "Too low. I don't like to take risks"
        },
        'q17': {
            'a': "I am willing to take more risk, expecting to achieve much higher than average returns.",
            'b': "I can accept reductions of my initial capital so my investments to bring me significant profits over time.",
            'c': "I desire steady income and some capital gains from my portfolio, which may fluctuate in losses/profits.",
            'd': "I wish to achieve a stable income during the years of the investment and I accept small ups and downs.",
            'e': "I wish to maintain the value of my original capital."
        },
        'q18': {
            'a': "Always the potential profits",
            'b': "Usually the potential profits",
            'c': "Both potential gains and potential losses",
            'd': "Usually the potential losses",
            'e': "Always potential losses"
        },
        'q19': {
            'a': "I would see this as an opportunity for significant new placements",
            'b': "I would see this as an opportunity for a little repositioning",
            'c': "I wouldn't do anything",
            'd': "I would liquidate a part of the investment",
            'e': "I would liquidate the entire investment"
        },
        'q13': {
            'a': "Above 1 million euros",
            'b': "300,001 to 1 million euros",
            'c': "80,001 to 300,000 euros",
            'd': "30,001 to 80,000 euros",
            'e': "Up to 30,000 euros"
        },
        'q6': {
            'a': "Low. It is not in my interests to be informed about financial news.",
            'b': "Average. I occasionally update on the main financial news.",
            'c': "Important. I regularly follow the news in the industry.",
            'd': "High. I am constantly informed about developments."
        },
        'q7': {
            'a': "No or minimal experience (Fixed deposits, Bonds, Cash Accounts)",
            'b': "Moderate experience (Bond Accounts, Short-term Products)",
            'c': "Significant experience (Shares, Equity Accounts)",
            'd': "Extensive experience (Derivatives, Structured Products)"
        },
        'q8': {
            'a': "Rarely (1-2 times a year)",
            'b': "Occasional (1 time every 2-3 months)",
            'c': "Often (1 time every fortnight or month)",
            'd': "Very often (at least 2 times a week)"
        }
    }
    
    # Display questions and collect responses
    for q_id, question in questions.items():
        st.subheader(question)
        response = st.radio(
            f"Select your answer for: {question}",
            options=list(options[q_id].keys()),
            format_func=lambda x: options[q_id][x],
            key=q_id
        )
        st.session_state.questionnaire_responses[q_id] = response
    
    # Process questionnaire and update profile
    if st.button("Submit Questionnaire"):
        risk_level, investment_capacity = process_questionnaire_responses(st.session_state.questionnaire_responses)
        customer_df = update_customer_profile(customer_id_input, risk_level, investment_capacity, customer_df)
        st.success(f"Profile updated! Your risk level is {risk_level} and investment capacity is {investment_capacity}")
    
    # Button trigger for recommendations
    if st.sidebar.button("Generate Recommendations"):
        st.write(f"Generating recommendations for customer: **{customer_id_input}**")
        
        # Get recommendations
        recs = hybrid_recommendation(customer_id_input, rating_matrix, pred_ratings, rating_df, asset_df, 
                                     customer_df, limit_prices_df, weights, top_n=int(N))
        
        # Display recommendations with detailed information
        st.write("### Top Recommendations")
        
        # Create a detailed recommendations dataframe
        rec_details = pd.DataFrame({
            'Score': recs,
            'Asset Name': asset_df.set_index('ISIN')['assetName'].loc[recs.index],
            'Category': asset_df.set_index('ISIN')['assetCategory'].loc[recs.index],
            'Subcategory': asset_df.set_index('ISIN')['assetSubCategory'].loc[recs.index],
            'Sector': asset_df.set_index('ISIN')['sector'].loc[recs.index],
            'Industry': asset_df.set_index('ISIN')['industry'].loc[recs.index],
            'Profitability': limit_prices_df.set_index('ISIN')['profitability'].loc[recs.index],
            'Current Price': limit_prices_df.set_index('ISIN')['priceMaxDate'].loc[recs.index]
        })
        
        # Format the display
        st.dataframe(rec_details.style.format({
            'Score': '{:.4f}',
            'Profitability': '{:.2%}',
            'Current Price': '€{:.2f}'
        }))
        
        # Calculate and display ROI@10 and nDCG@10
        roi = compute_roi_at_k(recs, limit_prices_df, k=10)
        ndcg = compute_ndcg_at_k(recs, test_df, k=10)
        
        st.write("### Recommendation Quality Metrics")
        if roi is not None:
            st.write(f"ROI@10: **{roi:.2%}**")
        if ndcg is not None:
            st.write(f"nDCG@10: **{ndcg:.4f}**")
    
    if eval_mode:
        st.write("### Evaluation Metrics (Leave-One-Out)")
        try:
            rmse = compute_rmse(pred_ratings, test_df)
            precision, recall = precision_recall_at_n(
                hybrid_recommendation, train_df, test_df,
                rating_matrix, rating_df, asset_df, customer_df, limit_prices_df,
                weights, pred_ratings, N
            )
            
            if rmse is not None:
                st.write(f"RMSE on held-out buys: **{rmse:.4f}**")
            else:
                st.write("No RMSE computed - insufficient test data")
                
            if precision is not None and recall is not None:
                st.write(f"Precision@{N}: **{precision:.4f}**, Recall@{N}: **{recall:.4f}**")
            else:
                st.write("No Precision/Recall computed - insufficient test data")
                
        except Exception as e:
            st.error(f"Error computing evaluation metrics: {str(e)}")

if __name__ == '__main__':
    main()
