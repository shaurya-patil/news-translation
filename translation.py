"""
Translation Module
Handles text translation using mBART model with chunking support
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from preprocessing import preprocess_text, detect_language, split_into_sentences

# Configuration
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024
CHUNK_SIZE = 500  # Characters per chunk

# Language names mapping
LANGUAGE_NAMES = {
    "en_XX": "English", "es_XX": "Spanish", "fr_XX": "French", "de_DE": "German",
    "it_IT": "Italian", "pt_XX": "Portuguese", "ru_RU": "Russian", "ja_XX": "Japanese",
    "ko_KR": "Korean", "zh_CN": "Chinese", "ar_AR": "Arabic", "hi_IN": "Hindi",
    "tr_TR": "Turkish", "vi_VN": "Vietnamese", "th_TH": "Thai", "nl_XX": "Dutch",
    "pl_PL": "Polish", "cs_CZ": "Czech", "ro_RO": "Romanian", "sv_SE": "Swedish",
    "uk_UA": "Ukrainian", "fa_IR": "Persian", "he_IL": "Hebrew", "id_ID": "Indonesian",
    "bn_IN": "Bengali", "ta_IN": "Tamil", "te_IN": "Telugu", "ur_PK": "Urdu"
}


class TranslationModel:
    """Wrapper class for mBART translation model"""
    
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        """
        Initialize the translation model
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to load model on ('cuda' or 'cpu')
        """
        print(f"Loading translation model: {model_name}")
        self.device = device
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        print(f"✓ Model loaded on {device}")
    
    def translate_chunk(self, text, src_lang, target_lang):
        """
        Translate a single chunk of text
        
        Args:
            text (str): Text to translate
            src_lang (str): Source language code
            target_lang (str): Target language code
        
        Returns:
            str: Translated text
        """
        if not text.strip():
            return text
        
        # Set source language
        self.tokenizer.src_lang = src_lang
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(self.device)
        
        # Generate translation
        forced_bos_token_id = self.tokenizer.lang_code_to_id[target_lang]
        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=MAX_LENGTH
        )
        
        # Decode
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return translated
    
    def translate(self, text, target_lang, preprocess=True):
        """
        Translate text with automatic chunking for long texts
        
        Args:
            text (str): Text to translate
            target_lang (str): Target language code
            preprocess (bool): Whether to preprocess text before translation
        
        Returns:
            dict: Dictionary containing translation results
        """
        # Preprocess if requested
        original_text = text
        if preprocess:
            text = preprocess_text(text)
        
        # Detect source language
        src_lang = detect_language(text)
        
        # Check if translation needed
        if src_lang == target_lang:
            return {
                'original': original_text,
                'translated': text,
                'source_lang': src_lang,
                'target_lang': target_lang,
                'preprocessed': preprocess
            }
        
        # Split into sentences for better chunking
        sentences = split_into_sentences(text)
        
        # Group sentences into chunks based on CHUNK_SIZE
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > CHUNK_SIZE and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)}...")
            translated = self.translate_chunk(chunk, src_lang, target_lang)
            translated_chunks.append(translated)
        
        # Combine translated chunks
        final_translation = ' '.join(translated_chunks)
        
        return {
            'original': original_text,
            'translated': final_translation,
            'source_lang': src_lang,
            'target_lang': target_lang,
            'num_chunks': len(chunks),
            'preprocessed': preprocess
        }


def batch_translate(texts, target_lang, model=None):
    """
    Translate multiple texts
    
    Args:
        texts (list): List of texts to translate
        target_lang (str): Target language code
        model (TranslationModel): Pre-loaded model (optional)
    
    Returns:
        list: List of translation results
    """
    if model is None:
        model = TranslationModel()
    
    results = []
    for i, text in enumerate(texts):
        print(f"\nTranslating text {i+1}/{len(texts)}")
        result = model.translate(text, target_lang)
        results.append(result)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("TRANSLATION MODULE - DEMO")
    print("="*70)
    
    # Initialize model
    translator = TranslationModel()
    
    # Test texts
    test_texts = [
        "The stock market crashed today, losing over 500 points.",
        "Scientists have discovered a new exoplanet that could harbor life.",
        "The government announced new economic policies for the upcoming year."
    ]
    
    target_languages = ["hi_IN", "es_XX", "fr_XX"]  # Hindi, Spanish, French
    
    print("\n" + "="*70)
    print("TRANSLATION EXAMPLES")
    print("="*70)
    
    for text in test_texts[:2]:  # Test first 2 texts
        print(f"\nOriginal (English): {text}")
        
        for target_lang in target_languages:
            result = translator.translate(text, target_lang)
            lang_name = LANGUAGE_NAMES[target_lang]
            print(f"\n{lang_name}: {result['translated']}")
            print(f"  Source: {result['source_lang']} | Target: {result['target_lang']}")
    
    print("\n" + "="*70)
    print("LONG TEXT CHUNKING TEST")
    print("="*70)
    
    long_text = " ".join(test_texts * 3)  # Create longer text
    print(f"\nLong text length: {len(long_text)} characters")
    
    result = translator.translate(long_text, "hi_IN")
    print(f"Number of chunks created: {result['num_chunks']}")
    print(f"Translated successfully: {len(result['translated'])} characters")
    print(f"\nTranslation preview: {result['translated'][:200]}...")
    
    print("\n" + "="*70)
    print("✓ Translation module ready!")
    print("="*70)