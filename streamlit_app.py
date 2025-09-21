import os
import re
import time
import pandas as pd
import numpy as np
import praw
import prawcore
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_deepseek_api_key')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'your_reddit_client_id')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'your_reddit_client_secret')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'your_user_agent')

# Initialize Reddit API client
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    reddit_available = True
except Exception as e:
    st.warning(f"Reddit API initialization failed: {e}")
    reddit_available = False

def get_reddit_comments(subreddit: str, stock_symbol: str, time_filter: str = 'week', limit: int = 100) -> pd.DataFrame:
    """Get Reddit comments using official Reddit API."""
    if not reddit_available:
        st.error("Reddit API not available")
        return pd.DataFrame()
    
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        
        # Map time filter to Reddit API
        time_filter_map = {
            'day': 'day',
            'week': 'week',
            'month': 'month',
            'year': 'year',
            'all': 'all'
        }
        
        # Search for posts mentioning the stock symbol
        search_query = f'{stock_symbol}'
        posts = subreddit_obj.search(
            search_query,
            time_filter=time_filter_map.get(time_filter, 'week'),
            limit=limit
        )
        
        comments_data = []
        
        for post in posts:
            try:
                post.comments.replace_more(limit=0)  # Remove "More Comments"
                for comment in post.comments.list()[:limit]:
                    if hasattr(comment, 'body') and comment.body:
                        comments_data.append({
                            'body': comment.body,
                            'score': comment.score,
                            'created_utc': datetime.fromtimestamp(comment.created_utc),
                            'post_title': post.title,
                            'post_id': post.id,
                            'comment_id': comment.id
                        })
            except Exception as e:
                continue
        
        return pd.DataFrame(comments_data)
        
    except Exception as e:
        st.error(f"Error fetching Reddit comments: {e}")
        return pd.DataFrame()

def get_reddit_comments_pushshift(subreddit: str, stock_symbol: str, after_days: int = 7, limit: int = 100) -> pd.DataFrame:
    """Get Reddit comments using Pushshift API as fallback."""
    try:
        end_time = int(time.time())
        start_time = end_time - (after_days * 24 * 3600)
        
        url = f"https://api.pushshift.io/reddit/search/comment/"
        params = {
            'subreddit': subreddit,
            'q': stock_symbol,
            'after': start_time,
            'before': end_time,
            'size': limit,
            'sort': 'desc',
            'sort_type': 'score'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        comments = data.get('data', [])
        
        if not comments:
            return pd.DataFrame()
        
        df = pd.DataFrame(comments)
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        
        # Select relevant columns
        columns_to_keep = ['body', 'score', 'created_utc', 'subreddit']
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        if available_columns:
            return df[available_columns]
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.warning(f"Pushshift API error: {e}")
        return pd.DataFrame()

def get_reddit_json_feed(subreddit: str, stock_symbol: str, limit: int = 100) -> pd.DataFrame:
    """Get Reddit comments using JSON feeds as fallback."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            'q': stock_symbol,
            'sort': 'new',
            'limit': limit,
            't': 'week'
        }
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        posts = data.get('data', {}).get('children', [])
        
        comments_data = []
        for post in posts:
            post_data = post.get('data', {})
            if 'selftext' in post_data and post_data['selftext']:
                comments_data.append({
                    'body': post_data['selftext'],
                    'score': post_data.get('score', 0),
                    'created_utc': datetime.fromtimestamp(post_data['created_utc']),
                    'post_title': post_data.get('title', ''),
                    'post_id': post_data['id']
                })
        
        return pd.DataFrame(comments_data)
        
    except Exception as e:
        st.warning(f"JSON feed error: {e}")
        return pd.DataFrame()

def get_reddit_web_scrape(subreddit: str, stock_symbol: str, limit: int = 50) -> pd.DataFrame:
    """Web scraping method as last resort."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/search/"
        params = {
            'q': stock_symbol,
            'sort': 'new',
            't': 'week',
            'limit': limit
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # This is a simplified version - in practice, you'd use BeautifulSoup
        # For now, return empty DataFrame
        return pd.DataFrame()
        
    except Exception as e:
        st.warning(f"Web scraping error: {e}")
        return pd.DataFrame()

def get_reddit_data_multi_source(subreddit: str, stock_symbol: str, time_filter: str = 'week', limit: int = 100) -> pd.DataFrame:
    """Try multiple data sources in order of preference."""
    sources = [
        lambda: get_reddit_comments(subreddit, stock_symbol, time_filter, limit),
        lambda: get_reddit_json_feed(subreddit, stock_symbol, limit),
        lambda: get_reddit_comments_pushshift(subreddit, stock_symbol, 7, limit),
        lambda: get_reddit_web_scrape(subreddit, stock_symbol, limit)
    ]
    
    for i, source_func in enumerate(sources):
        try:
            df = source_func()
            if not df.empty:
                st.success(f"Successfully retrieved data using {['Official API', 'JSON Feed', 'Pushshift', 'Web Scraping'][i]}")
                return df
        except Exception as e:
            continue
    
    st.error("Failed to retrieve data from all sources")
    return pd.DataFrame()

def analyze_sentiment(text: str, model: str = 'deepseek') -> float:
    """Analyze sentiment of a single text using LLM."""
    try:
        if model == 'deepseek' and DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != 'your_deepseek_api_key':
            return analyze_sentiment_deepseek(text)
        elif model == 'openai' and OPENAI_API_KEY and OPENAI_API_KEY != 'your_openai_api_key':
            return analyze_sentiment_openai(text)
        else:
            # Fallback to simple sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'buy', 'strong', 'growth', 'profit', 'gain']
            negative_words = ['bad', 'poor', 'terrible', 'negative', 'bearish', 'sell', 'weak', 'loss', 'decline', 'drop']
            
            text_lower = text.lower()
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment = (positive_score - negative_score) / total_words
            return max(-1.0, min(1.0, sentiment))
            
    except Exception as e:
        return 0.0

def analyze_sentiment_deepseek(text: str) -> float:
    """Analyze sentiment using DeepSeek API."""
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""
        Analyze the sentiment of this text about a stock investment:
        "{text}"
        
        Return ONLY a single number between -1.0 (extremely negative) and 1.0 (extremely positive).
        Do not include any explanation or additional text.
        """
        
        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 10,
            'temperature': 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        sentiment_text = result['choices'][0]['message']['content'].strip()
        
        try:
            sentiment = float(sentiment_text)
            return max(-1.0, min(1.0, sentiment))
        except ValueError:
            return 0.0
            
    except Exception as e:
        return 0.0

def analyze_sentiment_openai(text: str) -> float:
    """Analyze sentiment using OpenAI API."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""
        Analyze the sentiment of this text about a stock investment:
        "{text}"
        
        Return ONLY a single number between -1.0 (extremely negative) and 1.0 (extremely positive).
        Do not include any explanation or additional text.
        """
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 10,
            'temperature': 0.1
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        sentiment_text = result['choices'][0]['message']['content'].strip()
        
        try:
            sentiment = float(sentiment_text)
            return max(-1.0, min(1.0, sentiment))
        except ValueError:
            return 0.0
            
    except Exception as e:
        return 0.0

def analyze_sentiment_batch(texts: List[str], model: str = 'deepseek', show_api_info: bool = True) -> List[float]:
    """Analyze sentiment for a batch of texts."""
    sentiments = []
    for text in texts:
        sentiment = analyze_sentiment(text, model)
        sentiments.append(sentiment)
    return sentiments

def get_stock_data(stock_symbol: str, period: str = '1mo') -> Optional[Dict[str, Any]]:
    """Get stock data using yfinance."""
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[0]
        
        return {
            'current_price': current_price,
            'price_change': current_price - previous_price if current_price and previous_price else 0,
            'percent_change': ((current_price - previous_price) / previous_price * 100) if current_price and previous_price else 0,
            'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
            'high': hist['High'].max(),
            'low': hist['Low'].min(),
            'price_data': hist
        }
        
    except Exception as e:
        st.warning(f"Error fetching stock data: {e}")
        return None

def generate_investment_advice(stock_symbol: str, df: pd.DataFrame, stock_data: Optional[Dict[str, Any]], model: str = 'deepseek') -> str:
    """Generate investment advice based on sentiment analysis and stock data."""
    try:
        avg_sentiment = df['sentiment'].mean()
        sentiment_volatility = df['sentiment'].std()
        positive_ratio = (df['sentiment'] > 0.1).sum() / len(df)
        negative_ratio = (df['sentiment'] < -0.1).sum() / len(df)
        
        # Get most positive and negative comments
        most_positive = df.loc[df['sentiment'].idxmax()]
        most_negative = df.loc[df['sentiment'].idxmin()]
        
        # Stock data summary
        stock_summary = ""
        if stock_data:
            stock_summary = f"""
Stock Performance Summary:
- Current Price: ${stock_data['current_price']:.2f}
- Price Change: ${stock_data['price_change']:.2f} ({stock_data['percent_change']:.2f}%)
- Volume: {stock_data['volume']:,.0f}
- High/Low: ${stock_data['high']:.2f}/${stock_data['low']:.2f}
            """
        
        prompt = f"""
Based on the following Reddit sentiment analysis for {stock_symbol}, provide comprehensive investment advice:

Sentiment Analysis Results:
- Average Sentiment Score: {avg_sentiment:.3f} (-1.0 to 1.0 scale)
- Sentiment Volatility: {sentiment_volatility:.3f}
- Positive Comments: {positive_ratio:.1%}
- Negative Comments: {negative_ratio:.1%}
- Total Comments Analyzed: {len(df)}

{stock_summary}

Most Positive Comment:
"{most_positive['body'][:200]}..." (Score: {most_positive['sentiment']:.3f})

Most Negative Comment:
"{most_negative['body'][:200]}..." (Score: {most_negative['sentiment']:.3f})

Please provide:
1. Overall sentiment assessment
2. Investment recommendation (Buy/Hold/Sell)
3. Risk level assessment
4. Key factors to consider
5. Timeframe recommendation

Keep the advice concise and actionable.
        """
        
        if model == 'deepseek' and DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != 'your_deepseek_api_key':
            return generate_advice_deepseek(prompt)
        elif model == 'openai' and OPENAI_API_KEY and OPENAI_API_KEY != 'your_openai_api_key':
            return generate_advice_openai(prompt)
        else:
            return generate_simple_advice(avg_sentiment, positive_ratio, negative_ratio, stock_data)
            
    except Exception as e:
        return f"Error generating investment advice: {str(e)}"

def generate_advice_deepseek(prompt: str) -> str:
    """Generate advice using DeepSeek API."""
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        return f"DeepSeek API error: {str(e)}"

def generate_advice_openai(prompt: str) -> str:
    """Generate advice using OpenAI API."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        return f"OpenAI API error: {str(e)}"

def generate_simple_advice(avg_sentiment: float, positive_ratio: float, negative_ratio: float, stock_data: Optional[Dict[str, Any]]) -> str:
    """Generate simple investment advice when API is not available."""
    advice = []
    
    if avg_sentiment > 0.3:
        advice.append("**Strong Buy Signal**: Community sentiment is highly positive")
    elif avg_sentiment > 0.1:
        advice.append("**Buy Signal**: Community sentiment is moderately positive")
    elif avg_sentiment < -0.3:
        advice.append("**Strong Sell Signal**: Community sentiment is highly negative")
    elif avg_sentiment < -0.1:
        advice.append("**Sell Signal**: Community sentiment is moderately negative")
    else:
        advice.append("**Hold Signal**: Community sentiment is neutral")
    
    advice.append(f"\n**Sentiment Breakdown:**")
    advice.append(f"- Positive comments: {positive_ratio:.1%}")
    advice.append(f"- Negative comments: {negative_ratio:.1%}")
    
    if stock_data:
        advice.append(f"\n**Stock Performance:**")
        advice.append(f"- Current price: ${stock_data['current_price']:.2f}")
        advice.append(f"- Price change: {stock_data['percent_change']:.2f}%")
    
    advice.append("\n**Risk Assessment:**")
    if abs(avg_sentiment) < 0.2:
        advice.append("- Low risk: Mixed sentiment suggests stable market")
    else:
        advice.append("- Medium risk: Strong sentiment suggests potential volatility")
    
    return "\n".join(advice)

def format_markdown_to_html(markdown_text: str) -> str:
    """Simple markdown to text converter for basic formatting."""
    # Convert headers
    html = re.sub(r'^### (.*?)$', r'### \1', markdown_text, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'## \1', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*?)$', r'# \1', html, flags=re.MULTILINE)
    
    # Convert bold text
    html = re.sub(r'\*\*(.*?)\*\*', r'**\1**', html)
    
    # Convert italic text
    html = re.sub(r'\*(.*?)\*', r'*\1*', html)
    
    return html

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'loading_status' not in st.session_state:
        st.session_state.loading_status = False
    
    # Simple hero section
    st.title("📈 StockSentimentPro")
    st.markdown("AI-Powered Stock Community Sentiment Analysis Platform")
    st.markdown("Analyze Reddit community discussions with advanced AI technology for professional stock investment insights")
    
    # Check API key configuration
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key":
        st.warning("⚠️ DeepSeek API key is not configured. Please add your API key in the .env file:\n```\nDEEPSEEK_API_KEY=your_actual_api_key_here\n```\nYou can get your API key from the DeepSeek website.")
    
    # Fixed LLM configuration
    llm_model = 'deepseek'
    
    # Simple sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")
        st.markdown("Customize your analysis parameters")
        
        # Data source selection
        st.subheader("🔗 Data Source Selection")
        data_source_options = {
            'Smart Mode (Recommended)': 'auto',
            'Reddit Official API': 'official', 
            'JSON Data Feed': 'json_feeds',
            'Historical Archive': 'pushshift',
            'Web Scraping': 'web_scraping'
        }
        selected_source = st.selectbox(
            'Choose Data Retrieval Method',
            list(data_source_options.keys()),
            index=0,
            help="Smart mode automatically tries multiple data sources until successful"
        )
        data_source = data_source_options[selected_source]
        
        # Data source information
        with st.expander("ℹ️ Data Source Information"):
            if data_source == 'official':
                st.info("🔑 **Official API**: Requires Reddit API credentials. Most reliable but has rate limits")
            elif data_source == 'json_feeds':
                st.info("🌐 **JSON Data Feed**: No authentication required. Limited data but always available")
            elif data_source == 'pushshift':
                st.info("📚 **Historical Archive**: Archive data service. Good for historical content")
            elif data_source == 'web_scraping':
                st.warning("🕷️ **Web Scraping**: Experimental method. Use as last resort")
            else:
                st.info("🔄 **Smart Mode**: Automatically tries all data sources until successful")
        
        # Weight configuration
        st.subheader("🎛️ Analysis Weight Configuration")
        weight_llm = st.slider(
            'AI Sentiment Analysis Weight', 
            0.0, 1.0, 0.7, 0.1,
            help="Adjust the weight of AI analysis results in the final sentiment score"
        )
        weight_community = 1 - weight_llm
        
        # Display weight visualization
        st.write(f"AI Weight: {weight_llm:.1f} | Community Weight: {weight_community:.1f}")
        
        # Subreddit selection
        st.subheader("📋 Community Selection")
        popular_subreddits = {
            'wallstreetbets (Popular)': 'wallstreetbets',
            'stocks': 'stocks',
            'investing': 'investing',
            'SecurityAnalysis': 'SecurityAnalysis',
            'StockMarket': 'StockMarket'
        }
        subreddit_choice = st.selectbox(
            'Choose Discussion Community',
            list(popular_subreddits.keys()),
            help="Select the Reddit community to analyze"
        )
        subreddit = popular_subreddits[subreddit_choice]
        
        # Custom subreddit option
        if st.checkbox("Use Custom Community"):
            subreddit = st.text_input('Enter Community Name', value=subreddit)
        
        # Stock selection
        st.subheader("📊 Stock Selection")
        popular_stocks = {
            'Technology': ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA', 'TSLA'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V'],
            'Retail': ['AMZN', 'WMT', 'TGT', 'COST', 'HD'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK']
        }
        
        stock_category = st.selectbox('Select Stock Category', ['Custom Input'] + list(popular_stocks.keys()))
        
        if stock_category == 'Custom Input':
            stock_symbol = st.text_input(
                'Enter Stock Symbol', 
                'AAPL',
                help="Enter the stock symbol you want to analyze, such as AAPL, TSLA etc."
            ).upper()
        else:
            stock_symbol = st.selectbox('Select Specific Stock', popular_stocks[stock_category])
        
        # Time range selection
        time_filter = st.selectbox(
            '⏰ Time Range',
            ['Last 24 Hours', 'Last Week', 'Last Month', 'Last Year']
        )
        time_filter_map = {
            'Last 24 Hours': 'day',
            'Last Week': 'week', 
            'Last Month': 'month',
            'Last Year': 'year'
        }
        time_filter = time_filter_map[time_filter]
        
        # Analysis summary
        st.subheader("🚀 Start Analysis")
        st.markdown(f"""
        **Analysis Configuration Summary:**
        - **Target Stock:** {stock_symbol}
        - **Analysis Community:** r/{subreddit}
        - **Time Range:** {time_filter}
        - **AI Weight:** {weight_llm:.1f}
        """)
        
        # Analysis button
        start_analysis = st.button('🔥 Start Smart Analysis')
        
        # Add spacing after button
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display loading status if analyzing
        if st.session_state.get('loading_status', False):
            st.info("Analyzing...")
    
    # Analysis logic (outside sidebar)
    if start_analysis:
        st.session_state.loading_status = True
        
        with st.spinner('Fetching Reddit comments...'):
            # Choose data source based on user selection
            if data_source == 'auto':
                df = get_reddit_data_multi_source(subreddit, stock_symbol, time_filter)
            elif data_source == 'official':
                df = get_reddit_comments(subreddit, stock_symbol, time_filter)
            elif data_source == 'json_feeds':
                df = get_reddit_json_feed(subreddit, stock_symbol, limit=100)
            elif data_source == 'pushshift':
                df = get_reddit_comments_pushshift(subreddit, stock_symbol, after_days=7, limit=100)
            elif data_source == 'web_scraping':
                df = get_reddit_web_scrape(subreddit, stock_symbol, limit=50)
            else:
                st.error("Invalid data source selected")
                df = pd.DataFrame()
            
            # Fetch stock data
            try:
                stock_data = get_stock_data(stock_symbol)
            except Exception as e:
                st.warning(f"Could not fetch stock data: {str(e)}")
                stock_data = None
            
            st.session_state.analysis_results = df
            st.session_state.stock_data = stock_data
            st.session_state.analysis_completed = True
            st.session_state.loading_status = False
    
    # Display analysis results if available
    if st.session_state.analysis_completed and st.session_state.analysis_results is not None:
        df = st.session_state.analysis_results
        stock_data = st.session_state.get('stock_data', None)
        
        if len(df) == 0:
            st.warning('No relevant comments found')
            return
        
        st.success(f'Successfully retrieved {len(df)} comments')
        
        # Format time column
        df['formatted_time'] = df['created_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display raw data
        st.subheader('Comment Data')
        
        # Display unanalyzed comment data
        temp_df = df[['formatted_time', 'body', 'score']].copy()
        temp_df = temp_df.rename(columns={
            'formatted_time': 'Comment Time',
            'body': 'Comment Content',
            'score': 'Comment Score'
        })
        st.dataframe(temp_df)
        
        # Sentiment analysis
        st.subheader('Sentiment Analysis Progress')
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize sentiment score column
        df['sentiment'] = 0.0
        total_comments = len(df)
        batch_size = 10
        
        # Show API info once at the beginning
        st.info(f"🤖 Starting sentiment analysis using {llm_model} model...")
        
        # Batch process comments
        all_comments = df['body'].tolist()
        api_errors = 0
        
        for i in range(0, total_comments, batch_size):
            current_batch = all_comments[i:i + batch_size]
            current_progress = (i + len(current_batch)) / total_comments
            progress_bar.progress(current_progress)
            status_text.text(f'Analyzing comments {i + 1} to {i + len(current_batch)} of {total_comments}...')
            
            # Only show API info for the first batch
            show_info = (i == 0)
            batch_sentiments = analyze_sentiment_batch(current_batch, model=llm_model, show_api_info=show_info)
            
            # Count API errors (if all sentiments are 0, likely an error)
            if all(s == 0 for s in batch_sentiments):
                api_errors += 1
                
            df.iloc[i:i + len(current_batch), df.columns.get_loc('sentiment')] = batch_sentiments
        
        # Show final API statistics
        total_batches = (total_comments + batch_size - 1) // batch_size
        successful_batches = total_batches - api_errors
        if successful_batches > 0:
            st.success(f"✅ Analysis complete: {successful_batches}/{total_batches} batches processed successfully")
        if api_errors > 0:
            st.warning(f"⚠️ Warning: {api_errors}/{total_batches} batches returned zero values (possible API issues)")
        
        # Calculate combined sentiment score
        # Normalize comment scores to range between -1 and 1
        max_score = df['score'].max() if df['score'].max() > 0 else 1
        min_score = df['score'].min() if df['score'].min() < 0 else -1
        
        # Avoid division by zero
        score_range = max(abs(max_score), abs(min_score))
        
        # Add stronger error handling to ensure normalized_score column calculation doesn't fail
        try:
            if score_range > 0:
                df['normalized_score'] = df['score'] / score_range
            else:
                df['normalized_score'] = pd.Series([0] * len(df))
            
            # Ensure normalized_score column is numeric type
            df['normalized_score'] = pd.to_numeric(df['normalized_score'], errors='coerce')
            # Replace NaN values with 0
            df['normalized_score'].fillna(0, inplace=True)
            # Limit normalized scores to range between -1 and 1
            df['normalized_score'] = df['normalized_score'].clip(-1, 1)
        except Exception as e:
            st.error(f"Error calculating community score: {str(e)}")
            # If calculation fails, create a column of all zeros
            df['normalized_score'] = pd.Series([0] * len(df))
        
        # Create combined sentiment score (LLM sentiment + community score influence)
        df['combined_sentiment'] = (df['sentiment'] * weight_llm) + (df['normalized_score'] * weight_community)
        
        status_text.text('Sentiment analysis completed!')
        st.success('All comments have been analyzed')
        
        # Display data with sentiment analysis results
        st.subheader('Comment Data (with Sentiment Analysis)')
        
        # Add search box
        search_term = st.text_input('Search Comments', '')
        
        # Filter data
        if search_term:
            df = df[df['body'].str.contains(search_term, case=False, na=False)]

        # Data validity check to avoid pagination out of bounds
        if len(df) == 0:
            st.warning('No comments match your criteria. Please adjust your filters.')
            return
        # Pagination settings
        rows_per_page = st.selectbox('Rows per page', [10, 20, 50, 100], index=1)
        total_pages = len(df) // rows_per_page + (1 if len(df) % rows_per_page > 0 else 0)
        if total_pages == 0:
            total_pages = 1
        current_page = st.number_input('Page', min_value=1, max_value=total_pages, value=1)
        
        # Calculate current page data range
        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        
        # Display pagination info
        st.write(f'Showing {start_idx + 1} to {end_idx} of {len(df)} entries')
        
        # Display current page data
        st.dataframe(
            df[['formatted_time', 'body', 'sentiment', 'normalized_score', 'combined_sentiment', 'score']]
            .sort_values('formatted_time', ascending=False)
            .iloc[start_idx:end_idx]
            .rename(columns={
                'formatted_time': 'Comment Time',
                'body': 'Comment Content',
                'sentiment': 'LLM Sentiment',
                'normalized_score': 'Normalized Score',
                'combined_sentiment': 'Combined Sentiment',
                'score': 'Comment Score'
            })
        )
        
        # Intelligent Sentiment Analysis Dashboard
        st.subheader('📊 Intelligent Sentiment Analysis Insights')
        
        # Check data validity first
        if df['sentiment'].isnull().all() or df['sentiment'].nunique() <= 1:
            st.error("🔑 Sentiment analysis data is invalid - please check API configuration")
            return
        
        # === 1. Core KPI Dashboard ===
        st.subheader("🎯 Core Metrics Overview")
        
        # Calculate core metrics
        avg_sentiment = df['sentiment'].mean()
        sentiment_volatility = df['sentiment'].std()
        positive_ratio = (df['sentiment'] > 0.1).sum() / len(df)
        negative_ratio = (df['sentiment'] < -0.1).sum() / len(df)
        
        # Create simple metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😟" if avg_sentiment < -0.1 else "😐"
            st.metric(f"Overall Sentiment {sentiment_emoji}", f"{avg_sentiment:.3f}")
        
        with col2:
            volatility_level = "High" if sentiment_volatility > 0.3 else "Medium" if sentiment_volatility > 0.15 else "Low"
            st.metric("Sentiment Volatility", f"{sentiment_volatility:.3f}")
        
        with col3:
            st.metric("Positive Sentiment Ratio", f"{positive_ratio:.1%}")
        
        with col4:
            st.metric("Negative Sentiment Ratio", f"{negative_ratio:.1%}")
        
        # === 2. Sentiment Insights Analysis ===
        st.subheader("🔍 Deep Sentiment Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("#### Extreme Sentiment Comments")
            
            # Most positive comment
            most_positive = df.loc[df['sentiment'].idxmax()]
            st.success(f"**Most Positive Comment** (Score: {most_positive['sentiment']:.3f})")
            st.caption(f"Community Score: {most_positive['score']} | Time: {most_positive['formatted_time']}")
            st.write(f"💬 {most_positive['body'][:200]}...")
            
            # Most negative comment
            most_negative = df.loc[df['sentiment'].idxmin()]
            st.error(f"**Most Negative Comment** (Score: {most_negative['sentiment']:.3f})")
            st.caption(f"Community Score: {most_negative['score']} | Time: {most_negative['formatted_time']}")
            st.write(f"💬 {most_negative['body'][:200]}...")
        
        with insight_col2:
            st.markdown("#### High Attention Comments")
            
            # Highest community score comment
            top_community = df.loc[df['score'].idxmax()]
            st.info(f"**Most Popular Community Comment** (Reddit Score: {top_community['score']})")
            st.caption(f"AI Sentiment Score: {top_community['sentiment']:.3f} | Time: {top_community['formatted_time']}")
            st.write(f"💬 {top_community['body'][:200]}...")
            
            # Calculate correlation between sentiment intensity and community attention
            correlation = df['sentiment'].abs().corr(df['score'])
            correlation_desc = "Strong" if abs(correlation) > 0.5 else "Moderate" if abs(correlation) > 0.3 else "Weak"
            st.metric("Sentiment Intensity vs Community Attention", f"{correlation:.3f} ({correlation_desc})")
        
        # === 3. Time Pattern Analysis ===
        st.subheader("⏰ Time Dimension Analysis")
        
        # Create time series analysis
        df_time = df.sort_values('created_utc')
        df_time['hour'] = df_time['created_utc'].dt.hour
        df_time['date'] = df_time['created_utc'].dt.date
        
        time_col1, time_col2 = st.columns(2)
        
        with time_col1:
            # Sentiment time trend
            if len(df_time['date'].unique()) > 1:
                daily_sentiment = df_time.groupby('date')['sentiment'].mean()
                
                # Ensure we have valid daily data
                if len(daily_sentiment) > 1:
                    fig_time = px.line(x=daily_sentiment.index, y=daily_sentiment.values,
                                     title='Sentiment Change Over Time', markers=True)
                    fig_time.update_layout(xaxis_title='Date', yaxis_title='Average Sentiment Score', height=350)
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("📊 Single day data - showing hourly analysis below")
            else:
                # If only one day of data, analyze by hour
                hourly_sentiment = df_time.groupby('hour')['sentiment'].mean()
                
                # Check if we have valid hourly data
                if len(hourly_sentiment) > 1:
                    # Use line chart for time trends
                    fig_hour = px.line(x=hourly_sentiment.index, y=hourly_sentiment.values,
                                    title='Sentiment Trend by Hour', markers=True)
                    fig_hour.update_layout(xaxis_title='Hour', yaxis_title='Average Sentiment Score', height=350)
                    st.plotly_chart(fig_hour, use_container_width=True)
                elif len(hourly_sentiment) == 1:
                    # Single time point - show as a simple metric
                    hour = hourly_sentiment.index[0]
                    sentiment = hourly_sentiment.values[0]
                    st.info(f"📊 All comments posted at hour {hour}, average sentiment: {sentiment:.3f}")
                else:
                    st.warning("⚠️ No valid time data available for temporal analysis")
        
        with time_col2:
            # Sentiment distribution heatmap
            sentiment_ranges = pd.cut(df['sentiment'], bins=[-1, -0.3, -0.1, 0.1, 0.3, 1], 
                                    labels=['Strong Negative', 'Weak Negative', 'Neutral', 'Weak Positive', 'Strong Positive'])
            sentiment_dist = sentiment_ranges.value_counts()
            
            colors = ['#ff4444', '#ff8888', '#cccccc', '#88ff88', '#44ff44']
            fig_dist = px.pie(values=sentiment_dist.values, names=sentiment_dist.index,
                            title='Sentiment Distribution Structure', color_discrete_sequence=colors)
            fig_dist.update_layout(height=350)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # === 4. Content Quality Analysis ===
        st.subheader("📝 Content Quality Analysis")
        
        # High quality comments (high sentiment score + high community score)
        df['quality_score'] = df['sentiment'].abs() * 0.7 + (df['score'] / df['score'].max()) * 0.3
        top_quality = df.nlargest(3, 'quality_score')
        
        st.markdown("#### 🏆 High Quality Comments (High Sentiment Intensity + High Community Recognition)")
        for i, (_, comment) in enumerate(top_quality.iterrows()):
            with st.expander(f"Quality Comment #{i+1} - Quality Score: {comment['quality_score']:.3f}"):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(comment['body'])
                with col_b:
                    st.metric("Sentiment Score", f"{comment['sentiment']:.3f}")
                    st.metric("Community Score", f"{comment['score']}")
                    st.caption(f"Posted: {comment['formatted_time']}")
        
        # === 5. Investment Signal Analysis ===
        st.subheader("🎯 Investment Signal Analysis")
        
        signal_col1, signal_col2, signal_col3 = st.columns(3)
        
        with signal_col1:
            # Sentiment polarization indicator
            polarization = (positive_ratio + negative_ratio) / (1 - (positive_ratio + negative_ratio) + 0.01)
            polarization_level = "High" if polarization > 1.5 else "Medium" if polarization > 0.8 else "Low"
            st.metric("Opinion Polarization Level", f"{polarization:.2f} ({polarization_level})")
        
        with signal_col2:
            # Sentiment consistency indicator  
            consistency = 1 - sentiment_volatility
            consistency_level = "High" if consistency > 0.7 else "Medium" if consistency > 0.5 else "Low"
            st.metric("Opinion Consistency", f"{consistency:.3f} ({consistency_level})")
        
        with signal_col3:
            # Comprehensive sentiment signal
            if avg_sentiment > 0.2 and consistency > 0.6:
                signal = "🟢 Strong Bullish"
            elif avg_sentiment > 0.1:
                signal = "🟡 Cautiously Bullish" 
            elif avg_sentiment < -0.2 and consistency > 0.6:
                signal = "🔴 Strong Bearish"
            elif avg_sentiment < -0.1:
                signal = "🟡 Cautiously Bearish"
            else:
                signal = "⚪ Neutral"
            
            st.metric("Overall Sentiment Signal", signal)
             
        # Enhanced Investment Advice Section

        st.subheader("💼 Smart Investment Advice")
        st.markdown("Professional investment insights and recommendations based on AI sentiment analysis and real-time stock data")
        
        # Display stock data section if available
        if stock_data is not None:
            st.success("📈 Stock Data Retrieved Successfully")
            st.markdown("Successfully obtained the latest stock price and trading data")
            
            # Display basic stock information
            st.subheader("📊 Stock Basic Information")
            stock_metrics = {
                "Current Price": f"${stock_data['current_price']:.2f}" if stock_data['current_price'] else "N/A",
                "Price Change": f"${stock_data['price_change']:.2f}" if stock_data['price_change'] else "N/A", 
                "Percentage Change": f"{stock_data['percent_change']:.2f}%" if stock_data['percent_change'] else "N/A",
                "Average Volume": f"{stock_data['volume']:,.0f}" if stock_data['volume'] else "N/A",
                "Period High": f"${stock_data['high']:.2f}" if stock_data['high'] else "N/A",
                "Period Low": f"${stock_data['low']:.2f}" if stock_data['low'] else "N/A"
            }
            
            cols = st.columns(3)
            for i, (metric, value) in enumerate(stock_metrics.items()):
                with cols[i % 3]:
                    st.metric(metric, value)
            
            # Stock price chart
            st.subheader("📈 Price Trend Chart")
            price_data = stock_data['price_data']
            price_df = price_data.reset_index()
            
            if not price_df.empty and 'Close' in price_df.columns:
                fig = px.line(
                    price_df, 
                    x='Date', 
                    y='Close',
                    title=f"{stock_symbol} Recent Price Trend",
                    labels={'Close': 'Closing Price ($)', 'Date': 'Date'},
                    markers=True
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10), 
                    height=400,
                    xaxis_title='Date',
                    yaxis_title='Closing Price ($)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Unable to display price chart data")
        else:
            st.warning("⚠️ Unable to retrieve stock data temporarily, will generate investment advice based on sentiment analysis")
            st.info("💡 Even without real-time stock prices, sentiment-based analysis still provides valuable reference value")
        
        # Generate investment advice regardless of stock data availability
        st.subheader("🤖 AI Investment Advice")
        
        try:
            with st.spinner('🧠 AI is analyzing data and generating investment advice...'):
                advice = generate_investment_advice(stock_symbol, df, stock_data, model=llm_model)
            
            # Display advice with simple styling
            st.markdown(advice)
            
            # Display analysis type indicator
            if stock_data:
                st.success("✅ Comprehensive Analysis: Based on sentiment data + stock data")
            else:
                st.info("ℹ️ Sentiment Analysis: Based on Reddit community discussions")
                
        except Exception as e:
            st.error(f"❌ Error generating investment advice: {str(e)}")
            st.warning("💡 Please try again later, or check API configuration")

if __name__ == '__main__':
    main()