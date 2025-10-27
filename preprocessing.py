"""
Text Preprocessing Module
Handles all text cleaning and normalization for NLP tasks
"""

import re
import contractions
import unicodedata
from langdetect import detect

# Language code mappings for mBART model
LANG_MAP = {
    "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX",
    "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN",
    "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT",
    "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO",
    "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", "zh-cn": "zh_CN",
    "zh": "zh_CN", "af": "af_ZA", "az": "az_AZ", "bn": "bn_IN", "fa": "fa_IR",
    "he": "he_IL", "hr": "hr_HR", "id": "id_ID", "ka": "ka_GE", "km": "km_KH",
    "mk": "mk_MK", "ml": "ml_IN", "mn": "mn_MN", "mr": "mr_IN", "pl": "pl_PL",
    "ps": "ps_AF", "pt": "pt_XX", "sv": "sv_SE", "sw": "sw_KE", "ta": "ta_IN",
    "te": "te_IN", "th": "th_TH", "tl": "tl_XX", "uk": "uk_UA", "ur": "ur_PK",
    "xh": "xh_ZA", "gl": "gl_ES", "sl": "sl_SI"
}


def expand_contractions(text):
    """
    Expand contractions in text
    Example: can't -> cannot, they'll -> they will
    """
    return contractions.fix(text)


def remove_urls(text):
    """Remove URLs from text"""
    return re.sub(r'http\S+|www\.\S+', '', text)


def remove_emails(text):
    """Remove email addresses from text"""
    return re.sub(r'\S+@\S+', '', text)


def remove_html_tags(text):
    """Remove HTML tags from text"""
    return re.sub(r'<.*?>', '', text)


def normalize_unicode(text):
    """
    Normalize unicode characters to ASCII
    Handles special characters and accents
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')


def remove_extra_whitespace(text):
    """Remove extra whitespace and normalize spacing"""
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(text, aggressive=False):
    """
    Complete preprocessing pipeline
    
    Args:
        text (str): Input text to preprocess
        aggressive (bool): If True, applies more aggressive cleaning
    
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Expand contractions
    text = expand_contractions(text)
    
    # Step 2: Remove URLs
    text = remove_urls(text)
    
    # Step 3: Remove emails
    text = remove_emails(text)
    
    # Step 4: Remove HTML tags
    text = remove_html_tags(text)
    
    # Step 5: Normalize unicode
    text = normalize_unicode(text)
    
    # Step 6: Remove extra whitespace
    text = remove_extra_whitespace(text)
    
    # Optional: Aggressive cleaning
    if aggressive:
        # Remove all punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
    
    return text


def detect_language(text, fallback="en_XX"):
    """
    Detect the language of text and return mBART language code
    
    Args:
        text (str): Input text
        fallback (str): Default language code if detection fails
    
    Returns:
        str: mBART language code (e.g., 'en_XX', 'hi_IN')
    """
    try:
        detected = detect(text)
        return LANG_MAP.get(detected, fallback)
    except Exception:
        return fallback


def split_into_sentences(text):
    """
    Split text into sentences
    Simple sentence boundary detection
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_text_statistics(text):
    """
    Get statistics about the text
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Dictionary containing text statistics
    """
    words = text.split()
    sentences = split_into_sentences(text)
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0
    }


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("TEXT PREPROCESSING MODULE - DEMO")
    print("="*70)
    
    # Test cases
    test_texts = [
        "I can't believe it! Visit http://example.com for more info.",
        "Scientists've discovered a new planet at info@nasa.gov",
        "The stock market crashed today... It's    not     good!",
        "<html>Breaking News:</html> Too many spaces here.",
        "Don't miss this! www.news.com has all the details."
    ]
    
    print("\n1. BASIC PREPROCESSING:")
    print("-" * 70)
    for text in test_texts:
        preprocessed = preprocess_text(text)
        print(f"Original:     {text}")
        print(f"Preprocessed: {preprocessed}\n")
    
    print("\n2. LANGUAGE DETECTION:")
    print("-" * 70)
    multilingual_texts = [
        ("Hello, how are you?", "English"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Hola, ¿cómo estás?", "Spanish"),
        ("नमस्ते, आप कैसे हैं?", "Hindi")
    ]
    
    for text, expected_lang in multilingual_texts:
        detected = detect_language(text)
        print(f"{expected_lang}: {text}")
        print(f"Detected code: {detected}\n")
    
    print("\n3. TEXT STATISTICS:")
    print("-" * 70)
    sample = "This is a sample text. It has multiple sentences! Let's analyze it."
    stats = get_text_statistics(sample)
    print(f"Text: {sample}\n")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*70)
    print("✓ Preprocessing module ready!")
    print("="*70)