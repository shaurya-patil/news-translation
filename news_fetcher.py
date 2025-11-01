"""
News Fetcher Module
Handles fetching news from NewsAPI with various filters
Fixed to work properly with all countries
"""

import requests
from datetime import datetime, timedelta

# Configuration
NEWS_API_KEY = "bd2b48063bdf4040a5122edf3cfd0f3a"
BASE_URL = "https://newsapi.org/v2/"

# Country to domain mapping for fallback
COUNTRY_DOMAINS = {
    "us": "cnn.com,nytimes.com,washingtonpost.com,reuters.com,apnews.com",
    "gb": "bbc.co.uk,theguardian.com,telegraph.co.uk,independent.co.uk",
    "ca": "cbc.ca,theglobeandmail.com,nationalpost.com",
    "au": "abc.net.au,smh.com.au,theaustralian.com.au",
    "in": "timesofindia.indiatimes.com,hindustantimes.com,indianexpress.com",
    "de": "spiegel.de,faz.net,sueddeutsche.de,welt.de",
    "fr": "lemonde.fr,lefigaro.fr,liberation.fr",
    "it": "corriere.it,repubblica.it,lastampa.it",
    "jp": "asahi.com,mainichi.jp,yomiuri.co.jp",
    "kr": "chosun.com,joins.com,donga.com"
}


class NewsFetcher:
    """Class to fetch news from NewsAPI"""
    
    def __init__(self, api_key=NEWS_API_KEY):
        """
        Initialize NewsFetcher
        
        Args:
            api_key (str): NewsAPI key
        """
        self.api_key = api_key
        self.base_url = BASE_URL
    
    def get_top_headlines(self, country="us", category=None, query=None, page_size=10):
        """
        Fetch top headlines with automatic fallback
        
        Args:
            country (str): Country code (us, gb, in, etc.)
            category (str): Category (business, tech, sports, etc.)
            query (str): Search keyword
            page_size (int): Number of articles to fetch
        
        Returns:
            dict: API response with articles
        """
        # Try primary method first
        result = self._try_top_headlines(country, category, query, page_size)
        
        # If failed or no articles, try fallback
        if result.get('status') != 'ok' or not result.get('articles'):
            print(f"Primary method failed for {country}, trying fallback...")
            result = self._try_everything_fallback(country, category, query, page_size)
        
        return result
    
    def _try_top_headlines(self, country, category, query, page_size):
        """
        Try fetching from /top-headlines endpoint
        
        NOTE: NewsAPI restriction - cannot use 'country' with 'language' parameter
        """
        endpoint = f"{self.base_url}/top-headlines"
        
        params = {
            "apiKey": self.api_key,
            "pageSize": page_size
        }
        
        # Add country (this automatically determines language)
        if country:
            params["country"] = country
        
        # Add category
        if category and category != "all":
            params["category"] = category
        
        # Add search query
        if query:
            params["q"] = query
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check if we got valid articles
            if data.get('status') == 'ok' and data.get('articles'):
                return data
            
            return data
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f"Request failed: {str(e)}",
                'articles': []
            }
    
    def _try_everything_fallback(self, country, category, query, page_size):
        """
        Fallback to /everything endpoint with domain filtering
        
        This works better for countries where /top-headlines has limited coverage
        """
        endpoint = f"{self.base_url}/everything"
        
        params = {
            "apiKey": self.api_key,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "language": "en"  # Can use language in /everything
        }
        
        # Add domain filtering for specific countries
        if country in COUNTRY_DOMAINS:
            params["domains"] = COUNTRY_DOMAINS[country]
        
        # Build query
        query_parts = []
        
        if query:
            query_parts.append(query)
        
        if category and category != "all":
            query_parts.append(category)
        
        # Add country name to query if no specific query
        if not query:
            country_names = {
                "us": "United States OR America OR US",
                "gb": "United Kingdom OR Britain OR UK",
                "ca": "Canada OR Canadian",
                "au": "Australia OR Australian",
                "in": "India OR Indian",
                "de": "Germany OR German",
                "fr": "France OR French",
                "it": "Italy OR Italian",
                "jp": "Japan OR Japanese",
                "kr": "Korea OR Korean"
            }
            if country in country_names:
                query_parts.append(country_names[country])
        
        if query_parts:
            params["q"] = " ".join(query_parts)
        else:
            # Default query if nothing specified
            params["q"] = "news"
        
        # Add date range (last 3 days)
        today = datetime.now()
        three_days_ago = today - timedelta(days=3)
        params["from"] = three_days_ago.strftime("%Y-%m-%d")
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f"Fallback failed: {str(e)}",
                'articles': []
            }
    
    def search_news(self, query, from_date=None, to_date=None, language="en", page_size=10):
        """
        Search for news articles
        
        Args:
            query (str): Search query
            from_date (str): Start date (YYYY-MM-DD)
            to_date (str): End date (YYYY-MM-DD)
            language (str): Language code
            page_size (int): Number of articles
        
        Returns:
            dict: API response with articles
        """
        endpoint = f"{self.base_url}/everything"
        
        params = {
            "apiKey": self.api_key,
            "q": query,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt"
        }
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': str(e),
                'articles': []
            }
    
    def get_sources(self, category=None, language="en", country=None):
        """
        Get available news sources
        
        Args:
            category (str): Category filter
            language (str): Language filter
            country (str): Country filter
        
        Returns:
            dict: Available sources
        """
        endpoint = f"{self.base_url}/sources"
        
        params = {
            "apiKey": self.api_key
        }
        
        if category:
            params["category"] = category
        if language:
            params["language"] = language
        if country:
            params["country"] = country
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': str(e),
                'sources': []
            }


def format_article(article):
    """
    Format article data for display
    
    Args:
        article (dict): Raw article data from API
    
    Returns:
        dict: Formatted article
    """
    return {
        'title': article.get('title', 'No title'),
        'description': article.get('description', ''),
        'content': article.get('content', ''),
        'full_text': f"{article.get('description', '')} {article.get('content', '')}".strip(),
        'source': article.get('source', {}).get('name', 'Unknown'),
        'author': article.get('author', 'Unknown'),
        'url': article.get('url', ''),
        'image_url': article.get('urlToImage', ''),
        'published_at': article.get('publishedAt', '')[:10] if article.get('publishedAt') else 'Unknown'
    }


def filter_articles(articles, min_length=50):
    """
    Filter articles based on content length
    
    Args:
        articles (list): List of articles
        min_length (int): Minimum content length
    
    Returns:
        list: Filtered articles
    """
    filtered = []
    for article in articles:
        full_text = f"{article.get('description', '')} {article.get('content', '')}".strip()
        if len(full_text) >= min_length:
            filtered.append(article)
    return filtered


def get_article_statistics(articles):
    """
    Get statistics about fetched articles
    
    Args:
        articles (list): List of articles
    
    Returns:
        dict: Statistics
    """
    if not articles:
        return {
            'total': 0,
            'sources': [],
            'avg_length': 0
        }
    
    sources = set()
    lengths = []
    
    for article in articles:
        sources.add(article.get('source', {}).get('name', 'Unknown'))
        full_text = f"{article.get('description', '')} {article.get('content', '')}".strip()
        lengths.append(len(full_text))
    
    return {
        'total': len(articles),
        'sources': list(sources),
        'source_count': len(sources),
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0
    }


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("NEWS FETCHER MODULE - MULTI-COUNTRY TEST")
    print("="*70)
    
    # Initialize fetcher
    fetcher = NewsFetcher()
    
    # Test multiple countries
    test_countries = [
        ("us", "United States"),
        ("gb", "United Kingdom"),
        ("in", "India"),
        ("jp", "Japan"),
        ("de", "Germany")
    ]
    
    print("\n" + "="*70)
    print("TESTING MULTIPLE COUNTRIES")
    print("="*70)
    
    for country_code, country_name in test_countries:
        print(f"\n--- {country_name} ({country_code}) ---")
        
        result = fetcher.get_top_headlines(
            country=country_code,
            category="technology",
            page_size=3
        )
        
        if result.get('status') == 'ok':
            articles = result.get('articles', [])
            print(f"✅ Found {len(articles)} articles")
            
            if articles:
                print(f"\nSample article:")
                article = format_article(articles[0])
                print(f"  Title: {article['title'][:60]}...")
                print(f"  Source: {article['source']}")
        else:
            print(f"❌ Failed: {result.get('message')}")
    
    print("\n" + "="*70)
    print("TESTING SEARCH WITH QUERY")
    print("="*70)
    
    result = fetcher.get_top_headlines(
        country="jp",
        query="technology",
        page_size=5
    )
    
    if result.get('status') == 'ok':
        articles = result.get('articles', [])
        print(f"✅ Found {len(articles)} articles about technology in Japan")
    else:
        print(f"⚠ Search result: {result.get('message')}")
    
    print("\n" + "="*70)
    print("✓ Multi-country news fetcher ready!")
    print("="*70)