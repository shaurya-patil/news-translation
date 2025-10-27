"""
News Fetcher Module
Handles fetching news from NewsAPI with various filters
"""

import requests
from datetime import datetime, timedelta

# Configuration
NEWS_API_KEY = "bd2b48063bdf4040a5122edf3cfd0f3a"
BASE_URL = "https://newsapi.org/v2"


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
        Fetch top headlines
        
        Args:
            country (str): Country code (us, gb, in, etc.)
            category (str): Category (business, tech, sports, etc.)
            query (str): Search keyword
            page_size (int): Number of articles to fetch
        
        Returns:
            dict: API response with articles
        """
        endpoint = f"{self.base_url}/top-headlines"
        
        params = {
            "apiKey": self.api_key,
            "pageSize": page_size,
            "language": "en"
        }
        
        if country:
            params["country"] = country
        
        if category and category != "all":
            params["category"] = category
        
        if query:
            params["q"] = query
        
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
    print("NEWS FETCHER MODULE - DEMO")
    print("="*70)
    
    # Initialize fetcher
    fetcher = NewsFetcher()
    
    print("\n" + "="*70)
    print("1. FETCHING TOP HEADLINES (US - Technology)")
    print("="*70)
    
    result = fetcher.get_top_headlines(
        country="us",
        category="technology",
        page_size=5
    )
    
    if result.get('status') == 'ok':
        articles = result.get('articles', [])
        print(f"\nFound {len(articles)} articles")
        
        for i, article in enumerate(articles[:3], 1):
            formatted = format_article(article)
            print(f"\n{i}. {formatted['title']}")
            print(f"   Source: {formatted['source']}")
            print(f"   Published: {formatted['published_at']}")
            print(f"   Content length: {len(formatted['full_text'])} chars")
    else:
        print(f"Error: {result.get('message')}")
    
    print("\n" + "="*70)
    print("2. SEARCHING NEWS (AI keyword)")
    print("="*70)
    
    # Search for AI-related news from last week
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    
    result = fetcher.search_news(
        query="artificial intelligence",
        from_date=week_ago.strftime("%Y-%m-%d"),
        to_date=today.strftime("%Y-%m-%d"),
        page_size=5
    )
    
    if result.get('status') == 'ok':
        articles = result.get('articles', [])
        print(f"\nFound {len(articles)} articles about AI")
        
        for i, article in enumerate(articles[:3], 1):
            formatted = format_article(article)
            print(f"\n{i}. {formatted['title']}")
            print(f"   Source: {formatted['source']}")
    else:
        print(f"Error: {result.get('message')}")
    
    print("\n" + "="*70)
    print("3. FILTERING ARTICLES")
    print("="*70)
    
    result = fetcher.get_top_headlines(country="us", page_size=10)
    
    if result.get('status') == 'ok':
        all_articles = result.get('articles', [])
        filtered = filter_articles(all_articles, min_length=100)
        
        print(f"\nTotal articles: {len(all_articles)}")
        print(f"After filtering (min 100 chars): {len(filtered)}")
    
    print("\n" + "="*70)
    print("4. ARTICLE STATISTICS")
    print("="*70)
    
    result = fetcher.get_top_headlines(country="us", page_size=10)
    
    if result.get('status') == 'ok':
        articles = result.get('articles', [])
        stats = get_article_statistics(articles)
        
        print(f"\nTotal articles: {stats['total']}")
        print(f"Number of sources: {stats['source_count']}")
        print(f"Average content length: {stats['avg_length']:.0f} chars")
        print(f"Min length: {stats['min_length']} chars")
        print(f"Max length: {stats['max_length']} chars")
        print(f"\nSources: {', '.join(stats['sources'][:5])}")
    
    print("\n" + "="*70)
    print("5. GETTING AVAILABLE SOURCES")
    print("="*70)
    
    result = fetcher.get_sources(category="technology", language="en")
    
    if result.get('status') == 'ok':
        sources = result.get('sources', [])
        print(f"\nFound {len(sources)} technology news sources")
        
        for i, source in enumerate(sources[:5], 1):
            print(f"{i}. {source.get('name')} - {source.get('description', '')[:50]}...")
    else:
        print(f"Error: {result.get('message')}")
    
    print("\n" + "="*70)
    print("âœ“ News fetcher module ready!")
    print("="*70)