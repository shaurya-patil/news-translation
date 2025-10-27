"""
Streamlit News Translator App
Fetches news from NewsAPI and translates to any language using mBART model
"""

import streamlit as st
import requests
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langdetect import detect
import torch
import re
from datetime import datetime, timedelta

# ---------- Config ----------
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 1024
CHUNK_SIZE = 500
NEWS_API_KEY = "bd2b48063bdf4040a5122edf3cfd0f3a"  # Get from https://newsapi.org
# ----------------------------

# Language mappings
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

LANGUAGES = {
    "en_XX": "English", "es_XX": "Spanish", "fr_XX": "French", "de_DE": "German",
    "it_IT": "Italian", "pt_XX": "Portuguese", "ru_RU": "Russian", "ja_XX": "Japanese",
    "ko_KR": "Korean", "zh_CN": "Chinese", "ar_AR": "Arabic", "hi_IN": "Hindi",
    "tr_TR": "Turkish", "vi_VN": "Vietnamese", "th_TH": "Thai", "nl_XX": "Dutch",
    "pl_PL": "Polish", "cs_CZ": "Czech", "ro_RO": "Romanian", "sv_SE": "Swedish",
    "uk_UA": "Ukrainian", "fa_IR": "Persian", "he_IL": "Hebrew", "id_ID": "Indonesian",
    "bn_IN": "Bengali", "ta_IN": "Tamil", "te_IN": "Telugu", "ur_PK": "Urdu"
}

@st.cache_resource
def load_model():
    """Load translation model (cached)"""
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

def detect_lang(text, fallback="en_XX"):
    """Detect language of text"""
    try:
        short = detect(text)
        return LANG_MAP.get(short, fallback)
    except Exception:
        return fallback

def split_into_sentences(text):
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]

def translate_chunk(text, src_lang, target_lang, tokenizer, model):
    """Translate a single chunk"""
    if not text.strip():
        return text
    
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    forced_id = tokenizer.lang_code_to_id[target_lang]
    generated = model.generate(**encoded, forced_bos_token_id=forced_id, max_length=MAX_LEN)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def translate_text(text, target_lang, tokenizer, model):
    """Translate text with chunking"""
    src_lang = detect_lang(text)
    
    if src_lang == target_lang:
        return text
    
    sentences = split_into_sentences(text)
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
    
    translated_chunks = []
    for chunk in chunks:
        translated = translate_chunk(chunk, src_lang, target_lang, tokenizer, model)
        translated_chunks.append(translated)
    
    return ' '.join(translated_chunks)

def fetch_news(query="", category="general", country="us", page_size=10):
    """Fetch news from NewsAPI"""
    base_url = "https://newsapi.org/v2/top-headlines"
    
    params = {
        "apiKey": NEWS_API_KEY,
        "pageSize": page_size,
        "language": "en"
    }
    
    if query:
        params["q"] = query
    if category and category != "all":
        params["category"] = category
    if country:
        params["country"] = country
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="News Translator", page_icon="🌐", layout="wide")

st.title("🌐 News Translator")
st.markdown("*Get the latest news and translate to any language*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key_input = st.text_input(
        "NewsAPI Key",
        value=NEWS_API_KEY,
        type="password",
        help="Get your free API key from https://newsapi.org"
    )
    if api_key_input:
        NEWS_API_KEY = api_key_input
    
    st.divider()
    
    target_lang = st.selectbox(
        "Translate to:",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=0
    )
    
    st.divider()
    
    category = st.selectbox(
        "Category:",
        ["all", "business", "entertainment", "general", "health", "science", "sports", "technology"]
    )
    
    country = st.selectbox(
        "Country:",
        ["us", "gb", "ca", "au", "in", "de", "fr", "it", "jp", "kr"],
        format_func=lambda x: {
            "us": "🇺🇸 United States", "gb": "🇬🇧 United Kingdom",
            "ca": "🇨🇦 Canada", "au": "🇦🇺 Australia", "in": "🇮🇳 India",
            "de": "🇩🇪 Germany", "fr": "🇫🇷 France", "it": "🇮🇹 Italy",
            "jp": "🇯🇵 Japan", "kr": "🇰🇷 South Korea"
        }[x]
    )
    
    search_query = st.text_input("Search keyword (optional):")
    
    if st.button("🔄 Fetch News", type="primary", use_container_width=True):
        st.session_state.fetch_news = True

# Load model only when needed
if 'model_loaded' not in st.session_state:
    with st.spinner("Loading translation model..."):
        tokenizer, model = load_model()
        st.session_state.model_loaded = True
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
    st.success(f"✅ Model loaded on {DEVICE}")
else:
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model

# Fetch and display news
if st.session_state.get('fetch_news', False):
    with st.spinner("Fetching news..."):
        news_data = fetch_news(
            query=search_query,
            category=category,
            country=country,
            page_size=10
        )
    
    if news_data and news_data.get('articles'):
        articles = news_data['articles']
        st.info(f"Found {len(articles)} articles")
        
        for idx, article in enumerate(articles):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"{idx + 1}. {article.get('title', 'No title')}")
                
                with col2:
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], width=150)
                
                # Original content
                with st.expander("📰 Original Article", expanded=True):
                    st.markdown(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                    st.markdown(f"**Published:** {article.get('publishedAt', 'Unknown')[:10]}")
                    
                    description = article.get('description', '')
                    content = article.get('content', '')
                    full_text = f"{description} {content}" if content else description
                    
                    st.write(full_text)
                    
                    if article.get('url'):
                        st.markdown(f"[Read full article]({article['url']})")
                
                # Translate button
                if st.button(f"🌐 Translate to {LANGUAGES[target_lang]}", key=f"translate_{idx}"):
                    with st.spinner("Translating..."):
                        translated = translate_text(full_text, target_lang, tokenizer, model)
                        st.session_state[f'translation_{idx}'] = translated
                
                # Show translation if available
                if f'translation_{idx}' in st.session_state:
                    with st.expander(f"🌍 Translation ({LANGUAGES[target_lang]})", expanded=True):
                        st.write(st.session_state[f'translation_{idx}'])
                
                st.divider()
    else:
        st.warning("No articles found. Try different search parameters.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by NewsAPI & mBART-50 | Get your API key at <a href='https://newsapi.org'>newsapi.org</a></small>
</div>
""", unsafe_allow_html=True)