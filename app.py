import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime, timedelta
import os

# Define the S&P 500 tickers (using a subset for demonstration)
SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 'XOM',
    'BAC', 'WMT', 'CVX', 'PFE', 'CSCO', 'ADBE', 'NKE', 'PEP', 'KO', 'DIS'
]

def load_stock_data():
    """Load real-time stock data using yfinance"""
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    for ticker in SP500_TICKERS:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get the latest day's OHLC data
            hist = stock.history(start=start_date, end=end_date)
            latest_price_data = hist.iloc[-1] if not hist.empty else pd.Series()
            
            data[ticker] = {
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'company_name': info.get('longName', ticker),
                'open': latest_price_data.get('Open', 0),
                'close': latest_price_data.get('Close', 0),
                'high': latest_price_data.get('High', 0),
                'low': latest_price_data.get('Low', 0)
            }
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {str(e)}")
            continue
    return data

def load_user_preferences():
    """Load user preferences from file or create new if doesn't exist"""
    if os.path.exists('user_preferences.pkl'):
        with open('user_preferences.pkl', 'rb') as f:
            return pickle.load(f)
    return {}

def save_user_preferences(preferences):
    """Save user preferences to file"""
    with open('user_preferences.pkl', 'wb') as f:
        pickle.dump(preferences, f)

def get_recommendations(user_id, user_preferences, n_recommendations=5):
    """Generate stock recommendations for a user using collaborative filtering"""
    # Create user-item matrix
    user_item_matrix = pd.DataFrame(user_preferences).fillna(0)
    
    if len(user_preferences) < 2:  # If there's only one user
        # Return highest rated stocks from other sectors
        user_ratings = user_preferences[user_id]
        rated_stocks = set(user_ratings.keys())
        all_stocks = set(SP500_TICKERS)
        unrated_stocks = list(all_stocks - rated_stocks)
        
        # Get sectors of highest rated stocks
        stock_data = load_stock_data()
        rated_sectors = {stock_data[stock]['sector'] for stock in rated_stocks 
                        if stock in stock_data and user_ratings[stock] >= 4}
        
        # Recommend stocks from different sectors
        recommendations = []
        for stock in unrated_stocks:
            if stock in stock_data:
                if stock_data[stock]['sector'] in rated_sectors:
                    recommendations.append((stock, 0.8))  # High confidence for same sector
                else:
                    recommendations.append((stock, 0.5))  # Lower confidence for different sectors
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # Calculate user similarities
    user_similarities = cosine_similarity(user_item_matrix.T)
    user_similarities_df = pd.DataFrame(
        user_similarities,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    
    # Get stocks the user hasn't rated or has rated low
    user_stocks = {k for k, v in user_preferences[user_id].items() if v >= 3}  # Only consider as rated if >= 3 stars
    all_stocks = set(SP500_TICKERS)
    stocks_to_recommend = all_stocks - user_stocks
    
    # Calculate predicted ratings
    predictions = {}
    for stock in stocks_to_recommend:
        weighted_sum = 0
        similarity_sum = 0
        
        for other_user in user_preferences:
            if other_user != user_id and stock in user_preferences[other_user]:
                similarity = user_similarities_df.loc[user_id, other_user]
                rating = user_preferences[other_user][stock]
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            predictions[stock] = weighted_sum / similarity_sum
    
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

def main():
    st.set_page_config(page_title="Stock Recommender", layout="wide")
    
    # Initialize session state
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = load_user_preferences()
    
    # Login/Register section
    if 'user_id' not in st.session_state:
        st.title("Stock Recommendation System - Login")
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username")
            if st.button("Login/Register"):
                st.session_state.user_id = username
                if username not in st.session_state.user_preferences:
                    st.session_state.user_preferences[username] = {}
                save_user_preferences(st.session_state.user_preferences)
                st.rerun()
        return

    # Main application after login
    st.title(f"Stock Recommendation System - Welcome {st.session_state.user_id}")
    
    # Load real stock data
    stock_data = load_stock_data()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Rate Stocks", "Get Recommendations"])
    
    # Tab 1: Rate Stocks
    with tab1:
        st.header("Rate Stocks")
        
        # Create a grid layout for stock ratings
        cols = st.columns(4)
        for idx, ticker in enumerate(SP500_TICKERS):
            if ticker in stock_data:
                with cols[idx % 4]:
                    st.subheader(f"{ticker}")
                    st.write(f"Company: {stock_data[ticker]['company_name']}")
                    st.write(f"Sector: {stock_data[ticker]['sector']}")
                    st.write(f"Open: ${stock_data[ticker]['open']:.2f}")
                    st.write(f"Close: ${stock_data[ticker]['close']:.2f}")
                    if stock_data[ticker]['close'] > stock_data[ticker]['open']:
                        st.markdown("📈 Up")
                    elif stock_data[ticker]['close'] < stock_data[ticker]['open']:
                        st.markdown("📉 Down")
                    else:
                        st.markdown("➡️ No Change")
                    
                    # Show current rating if exists
                    current_rating = st.session_state.user_preferences[st.session_state.user_id].get(ticker, 0)
                    new_rating = st.slider(
                        f"Your Rating",
                        min_value=0,
                        max_value=5,
                        value=int(current_rating),
                        key=f"rating_{ticker}"
                    )
                    
                    # Update rating if changed
                    if new_rating != current_rating:
                        if new_rating > 0:  # Only save ratings > 0
                            st.session_state.user_preferences[st.session_state.user_id][ticker] = new_rating
                        elif ticker in st.session_state.user_preferences[st.session_state.user_id]:
                            del st.session_state.user_preferences[st.session_state.user_id][ticker]
                        save_user_preferences(st.session_state.user_preferences)
    
    # Tab 2: Get Recommendations
    with tab2:
        st.header("Your Recommended Stocks")
        
        if len(st.session_state.user_preferences[st.session_state.user_id]) < 2:
            st.warning("Please rate at least 2 stocks to get recommendations.")
            return
        
        recommendations = get_recommendations(st.session_state.user_id, st.session_state.user_preferences)
        
        if not recommendations:
            st.info("No recommendations available at this time. Try rating different stocks!")
            return
        
        for stock, score in recommendations:
            if stock in stock_data:
                with st.expander(f"{stock} - {stock_data[stock]['company_name']} (Score: {score:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Sector: {stock_data[stock]['sector']}")
                        st.write(f"Market Cap: ${stock_data[stock]['market_cap']:,.0f}")
                        st.write(f"Open: ${stock_data[stock]['open']:.2f}")
                        st.write(f"Close: ${stock_data[stock]['close']:.2f}")
                    with col2:
                        st.write(f"Beta: {stock_data[stock]['beta']:.2f}")
                        st.write(f"Dividend Yield: {stock_data[stock]['dividend_yield']*100:.2f}%")
                        st.write(f"High: ${stock_data[stock]['high']:.2f}")
                        st.write(f"Low: ${stock_data[stock]['low']:.2f}")
    
    # Logout button
    if st.sidebar.button("Logout"):
        for key in ['user_id', 'user_preferences']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()