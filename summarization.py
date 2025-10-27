"""
Summarization Module
Handles text summarization using BART model
"""

import torch
from transformers import pipeline
from preprocessing import preprocess_text

# Configuration
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Summarizer:
    """Wrapper class for text summarization"""
    
    def __init__(self, model_name=SUMMARIZATION_MODEL, device=DEVICE):
        """
        Initialize the summarization model
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to use ('cuda' or 'cpu')
        """
        print(f"Loading summarization model: {model_name}")
        device_id = 0 if device == "cuda" else -1
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=device_id
        )
        self.device = device
        print(f"✓ Summarizer loaded on {device}")
    
    def summarize(self, text, max_length=130, min_length=30, preprocess=True):
        """
        Summarize text
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
            preprocess (bool): Whether to preprocess text
        
        Returns:
            dict: Dictionary containing summary results
        """
        if not text or not text.strip():
            return {
                'original': text,
                'summary': text,
                'error': 'Empty text provided'
            }
        
        original_text = text
        
        # Preprocess if requested
        if preprocess:
            text = preprocess_text(text)
        
        try:
            # Generate summary
            summary_output = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary = summary_output[0]['summary_text']
            
            return {
                'original': original_text,
                'summary': summary,
                'original_length': len(original_text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(original_text),
                'preprocessed': preprocess
            }
        
        except Exception as e:
            return {
                'original': original_text,
                'summary': None,
                'error': str(e)
            }
    
    def summarize_with_options(self, text, summary_type="short"):
        """
        Summarize with predefined length options
        
        Args:
            text (str): Text to summarize
            summary_type (str): 'short', 'medium', or 'long'
        
        Returns:
            dict: Summary results
        """
        length_configs = {
            'short': {'max_length': 50, 'min_length': 20},
            'medium': {'max_length': 130, 'min_length': 40},
            'long': {'max_length': 200, 'min_length': 80}
        }
        
        config = length_configs.get(summary_type, length_configs['medium'])
        return self.summarize(text, **config)


def batch_summarize(texts, summarizer=None, max_length=130, min_length=30):
    """
    Summarize multiple texts
    
    Args:
        texts (list): List of texts to summarize
        summarizer (Summarizer): Pre-loaded summarizer (optional)
        max_length (int): Maximum summary length
        min_length (int): Minimum summary length
    
    Returns:
        list: List of summary results
    """
    if summarizer is None:
        summarizer = Summarizer()
    
    results = []
    for i, text in enumerate(texts):
        print(f"Summarizing text {i+1}/{len(texts)}...")
        result = summarizer.summarize(text, max_length, min_length)
        results.append(result)
    
    return results


def summarize_and_translate(text, target_lang, summarizer, translator):
    """
    Summarize text then translate the summary
    
    Args:
        text (str): Text to summarize and translate
        target_lang (str): Target language code
        summarizer (Summarizer): Summarizer instance
        translator (TranslationModel): Translator instance
    
    Returns:
        dict: Combined results
    """
    # First summarize
    summary_result = summarizer.summarize(text)
    
    if summary_result.get('error'):
        return summary_result
    
    # Then translate the summary
    from translation import TranslationModel
    translation_result = translator.translate(summary_result['summary'], target_lang)
    
    return {
        'original': text,
        'summary_english': summary_result['summary'],
        'summary_translated': translation_result['translated'],
        'target_lang': target_lang,
        'compression_ratio': summary_result['compression_ratio']
    }


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("SUMMARIZATION MODULE - DEMO")
    print("="*70)
    
    # Initialize summarizer
    summarizer = Summarizer()
    
    # Test texts (news articles)
    test_articles = [
        """
        The stock market experienced significant volatility today as investors reacted 
        to new economic data. The Dow Jones Industrial Average fell by 500 points in 
        early trading before recovering slightly by the close. Analysts attribute the 
        decline to concerns about inflation and rising interest rates. Federal Reserve 
        officials have indicated they may implement additional rate hikes in the coming 
        months to combat persistent inflation. Market experts advise investors to remain 
        cautious and diversify their portfolios during this uncertain period.
        """,
        
        """
        Scientists at NASA have announced the discovery of a potentially habitable 
        exoplanet located approximately 100 light-years from Earth. The planet, 
        designated as Kepler-442b, orbits within its star's habitable zone where 
        conditions could support liquid water. Using data from the Kepler Space 
        Telescope and ground-based observations, researchers determined that the 
        planet has a rocky composition similar to Earth. While the discovery is 
        exciting, scientists caution that much more research is needed to determine 
        if the planet actually harbors life. Future missions may include sending 
        probes to study the planet's atmosphere more closely.
        """
    ]
    
    print("\n" + "="*70)
    print("BASIC SUMMARIZATION")
    print("="*70)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n--- Article {i} ---")
        result = summarizer.summarize(article.strip())
        
        print(f"\nOriginal ({result['original_length']} chars):")
        print(result['original'][:150] + "...")
        
        print(f"\nSummary ({result['summary_length']} chars):")
        print(result['summary'])
        
        print(f"\nCompression ratio: {result['compression_ratio']:.2%}")
    
    print("\n" + "="*70)
    print("DIFFERENT SUMMARY LENGTHS")
    print("="*70)
    
    article = test_articles[0].strip()
    
    for summary_type in ['short', 'medium', 'long']:
        result = summarizer.summarize_with_options(article, summary_type)
        print(f"\n{summary_type.upper()} Summary ({result['summary_length']} chars):")
        print(result['summary'])
    
    print("\n" + "="*70)
    print("BATCH SUMMARIZATION")
    print("="*70)
    
    short_texts = [
        "The government announced new economic policies today.",
        "A major tech company unveiled its latest smartphone model.",
        "Climate scientists warn of accelerating global warming trends."
    ]
    
    results = batch_summarize(short_texts, summarizer, max_length=50, min_length=20)
    
    for i, result in enumerate(results, 1):
        if result.get('summary'):
            print(f"\n{i}. Original: {result['original']}")
            print(f"   Summary: {result['summary']}")
    
    print("\n" + "="*70)
    print("✓ Summarization module ready!")
    print("="*70)