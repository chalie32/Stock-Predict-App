from typing import List
import requests
from datetime import datetime, timedelta
import os
from dateutil import parser
import pytz
import logging
from .. import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAPIException(Exception):
    pass

def get_news(ticker: str, company_name: str = None) -> List[dict]:
    """
    Fetch recent news related to a stock ticker using NewsAPI.
    
    Args:
        ticker (str): Stock ticker symbol
        company_name (str, optional): Company name for better search results
        
    Returns:
        List[dict]: List of news articles with title, url, published date, and description
    """
    logger.info(f"Attempting to fetch news for ticker: {ticker}, company: {company_name}")
    
    # Try to get API key from environment or config
    api_key = os.getenv('NEWS_API_KEY') or getattr(config, 'NEWS_API_KEY', None)
    if not api_key:
        logger.error("NEWS_API_KEY not found in environment or config")
        raise NewsAPIException("NewsAPI key not found. Please set NEWS_API_KEY environment variable or add it to config.py")
    
    # Calculate date range (last 7 days for latest news)
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=7)
    
    # Format dates for the API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Build the API URL
    base_url = "https://newsapi.org/v2/everything"
    
    # Simple query focusing on latest stock news
    query = f'"{ticker}"'
    if company_name:
        # Clean up company name
        clean_company = company_name.lower()
        for suffix in [' inc', ' corp', ' ltd', ' limited', ' llc', ',']:
            clean_company = clean_company.replace(suffix, '').strip()
        query = f'({query} OR "{clean_company}")'
    
    logger.info(f"Search query: {query}")
    
    # Parameters for the API request
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 30,  # Get latest 30 articles
        'apiKey': api_key
    }
    
    try:
        logger.info(f"Making request to NewsAPI...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'ok':
            error_msg = f"API Error: {data.get('message', 'Unknown error')}"
            logger.error(error_msg)
            raise NewsAPIException(error_msg)
        
        total_results = len(data.get('articles', []))
        logger.info(f"Received {total_results} articles from API")
        
        # Process and format the news articles
        articles = []
        seen_urls = set()  # To avoid duplicate articles
        
        for article in data.get('articles', []):
            # Skip articles without title or description
            if not article.get('title') or not article.get('description'):
                continue
                
            # Skip duplicate articles
            url = article['url']
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            try:
                # Parse the date and convert to local timezone
                pub_date = parser.parse(article['publishedAt'])
                local_date = pub_date.astimezone(datetime.now().astimezone().tzinfo)
                
                # Format the date for display
                formatted_date = local_date.strftime("%b %d, %Y %I:%M %p")
                
                articles.append({
                    'title': article['title'],
                    'url': article['url'],
                    'published_date': formatted_date,
                    'description': article['description'],
                    'source': article.get('source', {}).get('name', 'Unknown Source')
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing date for article: {e}")
                continue
        
        # Sort articles by date
        articles.sort(key=lambda x: parser.parse(x['published_date']), reverse=True)
        
        logger.info(f"Returning {len(articles)} processed articles")
        return articles
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch news: {str(e)}"
        logger.error(error_msg)
        raise NewsAPIException(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error while fetching news: {str(e)}"
        logger.error(error_msg)
        raise NewsAPIException(error_msg) 