"""
Article Fetcher - ACTUALLY WORKING VERSION
"""

import requests
from bs4 import BeautifulSoup
import re
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

class ArticleFetcher:
    """Fetch full article content from URLs"""
    
    def __init__(self, timeout=15):
        self.timeout = timeout
    
    def fetch_article(self, url):
        """Fetch full article content"""
        try:
            time.sleep(1)  # Polite delay
            
            response = requests.get(url, headers=HEADERS, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove junk
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 
                            'iframe', 'noscript', 'form', 'button', 'img', 'svg']):
                tag.decompose()
            
            # Get article content
            content = self._extract_content(soup)
            
            if not content or len(content) < 200:
                return {
                    'success': False,
                    'error': 'Could not extract article content',
                    'url': url
                }
            
            # Clean it
            content = self._clean_text(content)
            
            return {
                'success': True,
                'content': content,
                'url': url,
                'length': len(content),
                'word_count': len(content.split())
            }
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                error = "Site blocks automated access"
            elif e.response.status_code == 404:
                error = "Article not found"
            else:
                error = f"HTTP {e.response.status_code}"
            return {'success': False, 'error': error, 'url': url}
        
        except Exception as e:
            return {'success': False, 'error': f"Error: {str(e)[:50]}", 'url': url}
    
    def _extract_content(self, soup):
        """Extract article text"""
        # Try article tag
        article = soup.find('article')
        if article:
            text = self._get_paragraphs(article)
            if len(text) > 500:
                return text
        
        # Try common classes
        for selector in [
            {'class': re.compile(r'article.*body|post.*content|entry.*content|story.*body', re.I)},
            {'id': re.compile(r'article|content|post', re.I)},
        ]:
            elem = soup.find('div', selector)
            if elem:
                text = self._get_paragraphs(elem)
                if len(text) > 500:
                    return text
        
        # Try main tag
        main = soup.find('main')
        if main:
            text = self._get_paragraphs(main)
            if len(text) > 500:
                return text
        
        # Find div with most paragraphs
        best = ""
        for div in soup.find_all('div'):
            paras = div.find_all('p', recursive=False)
            if len(paras) >= 3:
                text = ' '.join([p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40])
                if len(text) > len(best):
                    best = text
        
        if len(best) > 500:
            return best
        
        # Last resort: all paragraphs
        paras = soup.find_all('p')
        if len(paras) >= 5:
            texts = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40]
            return ' '.join(texts)
        
        return ""
    
    def _get_paragraphs(self, elem):
        """Get paragraph text from element"""
        paras = elem.find_all('p')
        if paras:
            texts = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40]
            return ' '.join(texts)
        return elem.get_text(separator=' ', strip=True)
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove junk patterns
        patterns = [
            r'Continue reading.*',
            r'Read more.*',
            r'Subscribe.*',
            r'Sign up.*',
            r'Advertisement',
            r'Click here.*',
            r'\[.*?\]',
            r'Share this.*',
            r'Follow us.*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()


def fetch_full_article(url, timeout=15):
    """Helper function"""
    fetcher = ArticleFetcher(timeout)
    return fetcher.fetch_article(url)