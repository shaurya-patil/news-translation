"""
Translation Module - SEAMLESS MULTI-LANGUAGE SUPPORT
Automatically detects and uses fine-tuned model
Falls back gracefully to base model if needed
"""

import torch
import os
import json
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from preprocessing import preprocess_text, detect_language, split_into_sentences

# Configuration
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
FINETUNED_MODEL = "./mbart_multilang_news"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024
CHUNK_SIZE = 500

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
    """Seamless translation with automatic model detection"""
    
    def __init__(self, model_name=None, device=DEVICE, use_finetuned=True):
        """
        Initialize translation model with automatic fallback
        
        Args:
            model_name (str): Specific model path (optional)
            device (str): 'cuda' or 'cpu'
            use_finetuned (bool): Try fine-tuned model first
        """
        self.device = device
        self.is_finetuned = False
        self.trained_languages = []
        
        # Determine model to load
        if model_name:
            model_to_load = model_name
        elif use_finetuned and self._check_finetuned_exists():
            model_to_load = FINETUNED_MODEL
            self.is_finetuned = True
        else:
            model_to_load = BASE_MODEL
        
        # Load model
        self._load_model(model_to_load)
        
        # Load language config if fine-tuned
        if self.is_finetuned:
            self._load_language_config()
    
    def _check_finetuned_exists(self):
        """Check if fine-tuned model exists"""
        if not os.path.exists(FINETUNED_MODEL):
            return False
        # Check for either safetensors or pytorch_model.bin
        has_safetensors = os.path.exists(os.path.join(FINETUNED_MODEL, "model.safetensors"))
        has_pytorch = os.path.exists(os.path.join(FINETUNED_MODEL, "pytorch_model.bin"))
        return has_safetensors or has_pytorch
    
    def _load_model(self, model_path):
        """Load model with fallback"""
        try:
            print(f"Loading translation model: {model_path}")
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
            self.model = MBartForConditionalGeneration.from_pretrained(model_path).to(self.device)
            
            if self.is_finetuned:
                print(f"✅ Fine-tuned multi-language model loaded")
                print(f"   📍 Path: {model_path}")
                print(f"   🎓 Transfer learning applied")
            else:
                print(f"✅ Base mBART-50 model loaded")
            
            print(f"✅ Device: {self.device}")
        
        except Exception as e:
            if self.is_finetuned:
                print(f"⚠️  Fine-tuned model failed, falling back to base model...")
                print(f"   Error: {str(e)[:100]}")
                self.is_finetuned = False
                self._load_model(BASE_MODEL)
            else:
                raise Exception(f"Failed to load model: {e}")
    
    def _load_language_config(self):
        """Load language configuration from fine-tuned model"""
        config_file = os.path.join(FINETUNED_MODEL, "language_config.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.trained_languages = config.get('trained_languages', [])
                    
                    if self.trained_languages:
                        print(f"\n   📚 Trained on {len(self.trained_languages)} languages:")
                        for lang in self.trained_languages:
                            lang_name = LANGUAGE_NAMES.get(lang, lang)
                            print(f"      • {lang_name} ({lang})")
                        print(f"   📊 Total samples: {config.get('total_samples', 'N/A')}")
                        print(f"   🎓 Method: {config.get('training_method', 'Transfer Learning')}")
            except Exception as e:
                print(f"   ⚠️  Could not load config: {e}")
    
    def translate_chunk(self, text, src_lang, target_lang):
        """Translate a single chunk"""
        if not text.strip():
            return text
        
        self.tokenizer.src_lang = src_lang
        
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(self.device)
        
        forced_bos_token_id = self.tokenizer.lang_code_to_id[target_lang]
        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=MAX_LENGTH
        )
        
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return translated
    
    def translate(self, text, target_lang, preprocess=True):
        """
        Seamless translation with automatic optimization
        
        Args:
            text (str): Text to translate
            target_lang (str): Target language code
            preprocess (bool): Apply preprocessing
        
        Returns:
            dict: Translation results with metadata
        """
        original_text = text
        
        if preprocess:
            text = preprocess_text(text)
        
        src_lang = detect_language(text)
        
        # Check if translation needed
        if src_lang == target_lang:
            return {
                'original': original_text,
                'translated': text,
                'source_lang': src_lang,
                'target_lang': target_lang,
                'preprocessed': preprocess,
                'model_type': 'fine-tuned' if self.is_finetuned else 'base',
                'optimized': False
            }
        
        # Check if target language was in training (for fine-tuned model)
        is_optimized = self.is_finetuned and target_lang in self.trained_languages
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > CHUNK_SIZE and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Translate chunks
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"Translating chunk {i+1}/{len(chunks)}...")
            translated = self.translate_chunk(chunk, src_lang, target_lang)
            translated_chunks.append(translated)
        
        final_translation = ' '.join(translated_chunks)
        
        return {
            'original': original_text,
            'translated': final_translation,
            'source_lang': src_lang,
            'target_lang': target_lang,
            'num_chunks': len(chunks),
            'preprocessed': preprocess,
            'model_type': 'fine-tuned' if self.is_finetuned else 'base',
            'optimized': is_optimized,
            'optimization_note': f'✅ Optimized for {LANGUAGE_NAMES.get(target_lang, target_lang)}' if is_optimized else ''
        }
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'model_type': 'Fine-tuned Multi-Language' if self.is_finetuned else 'Base mBART-50',
            'is_finetuned': self.is_finetuned,
            'device': self.device,
            'status': '✅ Custom-trained model active' if self.is_finetuned else '⚠️ Using base pretrained model'
        }
        
        if self.is_finetuned:
            info['trained_languages'] = self.trained_languages
            info['num_languages'] = len(self.trained_languages)
            info['training_method'] = 'Transfer Learning on News Commentary'
        
        return info
    
    def is_language_optimized(self, lang_code):
        """Check if a language was specifically trained"""
        return self.is_finetuned and lang_code in self.trained_languages


def batch_translate(texts, target_lang, model=None):
    """Batch translation with progress tracking"""
    if model is None:
        model = TranslationModel()
    
    results = []
    total = len(texts)
    
    for i, text in enumerate(texts, 1):
        print(f"\nTranslating {i}/{total}...")
        result = model.translate(text, target_lang)
        results.append(result)
    
    return results


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("SEAMLESS MULTI-LANGUAGE TRANSLATION - DEMO")
    print("="*70)
    
    # Initialize model
    translator = TranslationModel()
    
    # Show model info
    info = translator.get_model_info()
    print("\n📊 MODEL INFORMATION")
    print("-"*70)
    for key, value in info.items():
        if key == 'trained_languages' and isinstance(value, list):
            print(f"{key}:")
            for lang in value:
                print(f"  • {LANGUAGE_NAMES.get(lang, lang)} ({lang})")
        else:
            print(f"{key}: {value}")
    
    # Test translation
    print("\n" + "="*70)
    print("TRANSLATION TEST")
    print("="*70)
    
    test_text = "The government announced new economic policies today."
    test_langs = ["hi_IN", "es_XX", "fr_XX", "de_DE", "ja_XX"]
    
    print(f"\nOriginal: {test_text}\n")
    
    for lang in test_langs:
        result = translator.translate(test_text, lang, preprocess=False)
        lang_name = LANGUAGE_NAMES[lang]
        
        print(f"{lang_name:12} → {result['translated']}")
        if result['optimized']:
            print(f"             {result['optimization_note']}")
    
    print("\n" + "="*70)
    print("✓ Seamless translation ready!")
    print("="*70)