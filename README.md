# Financial Asset Recommendation

## Overview

The asset recommender leverages a **hybrid recommendation pipeline** that integrates:
- **Collaborative Filtering (CF):** Uses customers' past buy transactions.
- **Content-Based Filtering (CB):** Uses asset features and profitability data.
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
- MiFID-aligned structure for risk profiling

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
   - Details: Contains profitability data (ROI), first/last dates, and extreme values for every asset.

---

## System Components

### 1. Data Loading & Preprocessing

- **CSV Loading:**  
  Load all dataset files assuming CSV formatting and UTF-8 encoding.

- **Preprocessing Transactions:**  
  - Filter to include only "Buy" transactions.
  - Sort by timestamp for proper train-test splitting.
  - Build a customer × asset rating matrix using transaction counts.

- **Train-Test Split:**  
  - Use leave-one-out split for evaluation.
  - For each user, hold out their last transaction as test data.

---

### 2. Recommendation Pipeline

#### A. Collaborative Filtering
- **Matrix Factorization:**  
  - Use Truncated SVD with 5 components.
  - Compute latent factors for users and assets.
  - _Output:_ Predicted ratings for customer-asset pairs.

#### B. Content-Based Filtering
- **Asset Profile Building:**  
  - One-hot encode categorical features (category, subcategory, sector, industry, market).
  - Include profitability as a numerical feature.
  - Handle missing values with appropriate defaults.
  - _Output:_ Feature matrix for all assets.

- **User Profile & Scoring:**  
  - Build user profile as mean of their purchased assets' features.
  - Compute cosine similarity between user profile and all assets.
  - Handle cold-start with neutral scores.
  - _Output:_ Content-based similarity scores.

#### C. Demographic Based Scoring
- **Risk Profile Matching:**  
  - Map customer risk levels and investment capacity to numeric scores.
  - Compute weighted similarity between user demographics and asset categories.
  - _Output:_ Demographic scores for assets.

#### D. Hybrid Scoring
- **Component Weights:**  
  - Allow dynamic adjustment of weights for each component.
  - Weights can be set independently (no sum constraint).
  - Default weights: CF (0.4), CB (0.3), Demographic (0.3).

- **Score Combination:**  
  - Normalize each component's scores to [0,1] range.
  - Apply weighted combination.
  - _Output:_ Final composite scores.

#### E. Recommendation Generation
- **Filtering and Ranking:**  
  - Remove previously purchased assets.
  - Rank remaining assets by composite score.
  - Select Top-N recommendations.

---

### 3. Evaluation & Frontend

#### A. Evaluation Metrics
- **RMSE:**  
  - Compute on held-out test transactions.
  - Handle edge cases and insufficient data.

- **Precision@N & Recall@N:**  
  - Evaluate recommendation quality at specified N.
  - Robust handling of edge cases and errors.

#### B. Streamlit Frontend
- **User Interface:**  
  - Customer selection dropdown.
  - Component weight sliders.
  - Top-N parameter setting.
  - Evaluation metrics toggle.

- **Risk Assessment:**  
  - Interactive questionnaire for risk profiling.
  - Questions on risk appetite, investment expectations.
  - Automatic profile updates.

- **Recommendation Display:**  
  - Detailed asset information.
  - Formatted scores and metrics.
  - Profitability and price information.

---

## Data Flow Diagram

```mermaid
graph LR
    A[Customer Information]
    B[Asset Information]
    C[Transactions]
    D[Limit Prices]
    
    A -->|Preprocessing| F(Customer Profile)
    B -->|Feature Encoding| G(Asset Features)
    C -->|Filter & Aggregate| H(Rating Matrix)
    D -->|Merge with B| G
    
    H -->|SVD| J(CF Scores)
    G -->|Cosine Similarity| K(CB Scores)
    F -->|Risk Matching| L(Demographic Scores)
    
    J --> M[Score Normalization]
    K --> M
    L --> M
    
    M -->|Weighted Combination| N(Final Scores)
    N -->|Rank & Filter| O(Top-N Recommendations)
    
    O --> P[Streamlit UI]
    P -->|Questionnaire| Q[Risk Assessment]
    Q -->|Update| F