# Financial Asset Recommendation

## Overview

The asset recommender leverages a **hybrid recommendation pipeline** that integrates:
- **Collaborative Filtering (CF):** Uses customers' past buy transactions.
- **Content-Based Filtering (CB):** Enriches asset features with metadata and profitability data.
- **Demographic Based Scoring:** Incorporates customer risk profiles and demographics.

A **Streamlit** frontend is provided to allow user interaction and parameter tuning.

---

## Dataset Source

This system is built upon the **FAR-Trans dataset**, a comprehensive financial asset recommendation dataset, provided by a European financial institution.

**Citation:**

> Sanz-Cruzado, J., Droukas, N., & McCreadie, R. (2024).  
> **FAR-Trans: An Investment Dataset for Financial Asset Recommendation.**  
> *IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys)*, Jeju, South Korea.  
> [arXiv:2407.08692](https://arxiv.org/abs/2407.08692)

**License:** CC-BY 4.0  
**Link:** [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)

The dataset includes:
- Customer demographics and investment profiles
- Detailed financial product metadata
- Historical transaction logs
- Time-series pricing and profitability data
- MiFID-aligned  structure for risk profiling

---

## Data Sources

1. **Customer Information**
   - File: `customer_information.csv`
   - Details: Contains customer identifiers, type, risk level, investment capacity, and timestamps.

2. **Asset Information**
   - File: `asset_information.csv`
   - Details: Contains ISIN, asset name, asset categories/subcategories, market identifier, sector, industry, and update timestamps.

3. **Transactions**
   - File: `transactions.csv`
   - Details: Contains customer transactions (Buy/Sell) with monetary values, units, channels, and market information.
   - Note: Preprocessed to use only "Buy" transactions as positive interaction signals.

4. **Limit Prices**
   - File: `limit_prices.csv`
   - Details: Contains profitability data (ROI), first/last dates, and extreme values for every asset. This info is used to derive asset performance metrics.

---

## System Components

### 1. Data Loading & Preprocessing

- **CSV Loading:**  
  Load all dataset files assuming CSV formatting and UTF-8 encoding.

- **Preprocessing Transactions:**  
  - Filter to include only "Buy" transactions.
  - Aggregate the transactions (e.g., count or sum of buy events) to build a **customer × asset rating matrix**.

- **Asset Feature Enrichment:**  
  Merge asset metadata with profitability (from limit prices) and perform one-hot encoding on categorical fields (e.g., assetCategory and assetSubCategory).

- **Data Processing:**  
  Map  answers (using defined answer-to-score mappings) into numerical features that describe the customer’s risk tolerance and investment preferences.

---

### 2. Recommendation Pipeline

#### A. Collaborative Filtering
- **Matrix Factorization:**  
  Use Truncated SVD on the rating matrix to compute latent factors (user and asset embeddings).  
  _Output:_ Predicted ratings for customer-asset pairs.

#### B. Content-Based Filtering
- **Asset Profile Building:**  
  - Construct asset feature vectors from categorical fields and normalized profitability.
  - Build customer profiles by averaging feature vectors of previously purchased assets.
  - Compute cosine similarity between the customer profile and asset feature vectors.
  _Output:_ Content-based similarity scores for assets.

#### C. Demographic Based Scoring
- ** Demographic Matching:**  
  - Use customer metadata (risk level, investment capacity) to target suitable asset classes.
  _Output:_ Demographic scores reflecting the closeness of asset risk profiles to the customer’s risk tolerance.

#### D. Hybrid Scoring
- **Weighted Combination:**  
  Normalize the scores from CF, CB, and Demographic components (typically to a 0–1 range).
  Use a weighted average to combine:
  - CF Score
  - Content-Based Score
  - Enhanced Demographic/ Score  
  _Output:_ A final composite score per asset.

#### E. Recommendation Generation
- **Filtering and Ranking:**  
  Remove assets the customer already purchased.
  Rank the remaining assets by the composite score.
  Select the Top-N assets to recommend based on user-specified parameters.

---

### 3. System Integration & Frontend

- **Streamlit Frontend:**  
  - Provides an interactive UI for selecting a customer, adjusting component weights (CF, CB, Demographic), and setting the Top-N recommendations.
  - Displays the recommended asset list along with additional asset details.

- **Deployment Considerations:**
  - The solution runs as a self-contained Python script (`app.py`).
  - Dependencies include: `pandas`, `numpy`, `scikit-learn`, `scipy`, `streamlit`.
  - Deployed locally or on a cloud server supporting Streamlit deployments.

---

## Data Flow Diagram

```mermaid
graph LR
    A[Customer Information]
    B[Asset Information]
    C[Transactions]
    D[Limit Prices]
    
    A -->|Preprocessing| F(Customer Profile)
    B -->|Merge & Encode| G(Asset Features)
    C -->|Filter & Aggregate| H(Rating Matrix)
    D -->|Merge with B| G
    
    H -->|Matrix Factorization| J(Collaborative Filtering Scores)
    G -->|Cosine Similarity| K(Content-Based Scores)
    F -->| Demographics| L(Demographic Scores)
    
    J --> M[Score Normalization]
    K --> M
    L --> M
    
    M -->|Weighted Hybrid Combination| N(Final Composite Scores)
    N -->|Rank & Filter| O(Top-N Recommendations)
    
    O --> P[Streamlit Frontend]
