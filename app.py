"""
Multilingual News Platform - FULLY WORKING VERSION
Real-time news with translation and summarization
"""

import streamlit as st
import re
from preprocessing import preprocess_text
from translation import TranslationModel, LANGUAGE_NAMES
from summarization import Summarizer
from news_fetcher import NewsFetcher
from article_fetcher import ArticleFetcher

# Page config
st.set_page_config(
    page_title="Multilingual News",
    page_icon="🌐",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .translated-content {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        line-height: 1.8;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None
if 'translations' not in st.session_state:
    st.session_state.translations = {}
if 'full_articles' not in st.session_state:
    st.session_state.full_articles = {}

# Load models (cached)
@st.cache_resource(show_spinner=False)
def load_models():
    translator = TranslationModel()
    summarizer = Summarizer()
    return translator, summarizer

# Initialize
news_fetcher = NewsFetcher()
article_fetcher = ArticleFetcher()

# Sidebar
with st.sidebar:
    st.header("⚙ Settings")
    
    st.subheader("🌍 Translation")
    target_lang = st.selectbox(
        "Language:",
        options=list(LANGUAGE_NAMES.keys()),
        format_func=lambda x: LANGUAGE_NAMES[x],
        index=list(LANGUAGE_NAMES.keys()).index("hi_IN")
    )
    
    st.subheader("📝 Summary")
    summary_length = st.select_slider(
        "Length:",
        options=["Short", "Medium", "Long"],
        value="Medium"
    )
    
    st.divider()
    
    st.subheader("📰 News")
    country = st.selectbox(
        "Country:",
        ["us", "gb", "ca", "au", "in", "de", "fr", "it", "jp", "kr"],
        format_func=lambda x: {
            "us": "🇺🇸 USA", "gb": "🇬🇧 UK", "ca": "🇨🇦 Canada",
            "au": "🇦🇺 Australia", "in": "🇮🇳 India", "de": "🇩🇪 Germany",
            "fr": "🇫🇷 France", "it": "🇮🇹 Italy", "jp": "🇯🇵 Japan", "kr": "🇰🇷 S.Korea"
        }[x]
    )
    
    category = st.selectbox(
        "Category:",
        ["general", "business", "entertainment", "health", "science", "sports", "technology"],
        format_func=lambda x: x.title()
    )
    
    search_query = st.text_input("🔍 Keywords:")
    num_articles = st.slider("Articles:", 5, 20, 10)
    
    st.divider()
    
    fetch_button = st.button("🔄 Fetch News", type="primary", use_container_width=True)
    
    if st.session_state.articles:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.articles = []
            st.session_state.selected_article = None
            st.session_state.translations = {}
            st.session_state.full_articles = {}
            st.rerun()

# Header
st.markdown('<h1 class="main-title">🌐 Multilingual News Platform</h1>', unsafe_allow_html=True)
st.caption("Real-time news with AI translation & summarization")

# Load models
with st.spinner("🔄 Loading AI models... (first time: 2-5 min)"):
    translator, summarizer = load_models()
    
st.success("✅ Models ready!")

# Fetch news
if fetch_button:
    with st.spinner("📡 Fetching news..."):
        news_data = news_fetcher.get_top_headlines(
            country=country,
            category=category,
            query=search_query if search_query else None,
            page_size=num_articles
        )
    
    if news_data.get('status') == 'ok':
        articles = news_data.get('articles', [])
        # Filter out articles without content
        st.session_state.articles = [a for a in articles if a.get('title') and (a.get('description') or a.get('content'))]
        st.session_state.selected_article = None
        st.session_state.translations = {}
        st.session_state.full_articles = {}
        st.success(f"✅ Fetched {len(st.session_state.articles)} articles!")
    else:
        st.error(f"❌ {news_data.get('message', 'Failed to fetch news')}")

# Main layout
if st.session_state.articles:
    col1, col2 = st.columns([1, 2])
    
    # Article list
    with col1:
        st.subheader("📰 Articles")
        
        for idx, article in enumerate(st.session_state.articles):
            is_selected = st.session_state.selected_article == idx
            
            if st.button(
                f"{'▶' if is_selected else '▷'} {idx + 1}",
                key=f"btn_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_article = idx
                st.rerun()
            
            # Preview with TRANSLATED title
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if article.get('urlToImage'):
                    st.image(article['urlToImage'], width=60)
            with col_b:
                # Translate title for preview
                preview_key = f"prev_{idx}_{target_lang}"
                if preview_key not in st.session_state.translations:
                    title = article.get('title', 'No title')
                    try:
                        result = translator.translate(title[:100], target_lang, preprocess=False)
                        st.session_state.translations[preview_key] = result['translated'][:60]
                    except:
                        st.session_state.translations[preview_key] = title[:60]
                
                st.caption(f"{st.session_state.translations[preview_key]}...")
                st.caption(f"📅 {article.get('publishedAt', '')[:10]}")
            
            st.divider()
    
    # Article detail
    with col2:
        if st.session_state.selected_article is not None:
            article = st.session_state.articles[st.session_state.selected_article]
            idx = st.session_state.selected_article
            
            # Image
            if article.get('urlToImage'):
                st.image(article['urlToImage'], use_column_width=True)
            
            # Translated title
            title_key = f"t_{idx}_{target_lang}"
            if title_key not in st.session_state.translations:
                with st.spinner("Translating title..."):
                    title = article.get('title', '')
                    result = translator.translate(title, target_lang, preprocess=False)
                    st.session_state.translations[title_key] = result['translated']
            
            st.markdown(f"## {st.session_state.translations[title_key]}")
            st.caption(f"📰 {article.get('source', {}).get('name', 'Unknown')} • 📅 {article.get('publishedAt', '')[:10]}")
            
            if article.get('url'):
                st.markdown(f"[🔗 Read Original]({article['url']})")
            
            st.divider()
            
            # Summary
            st.subheader("📝 Summary")
            
            summary_key = f"s_{idx}{target_lang}{summary_length}"
            
            if summary_key not in st.session_state.translations:
                with st.spinner("Generating summary..."):
                    # Get content
                    desc = article.get('description', '') or ''
                    cont = article.get('content', '') or ''
                    full_text = f"{desc} {cont}".strip()
                    
                    # CLEAN THE TEXT PROPERLY
                    # Remove URLs
                    full_text = re.sub(r'https?://\S+', '', full_text)
                    full_text = re.sub(r'www\.\S+', '', full_text)
                    # Remove [+xxx chars] suffix
                    full_text = re.sub(r'\s*\[\+\d+\s+chars\]', '', full_text)
                    # Remove extra whitespace
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    
                    if len(full_text) > 150:
                        # Summarize
                        # Adaptive summary lengths based on article size
                        content_length = len(full_text.split())

                        def get_summary_lengths(length_type, words):
                            if length_type.lower() == "short":
                                return {
                                    "max_length": max(60, int(words * 0.4)),
                                    "min_length": max(30, int(words * 0.2))
                                }
                            elif length_type.lower() == "medium":
                                return {
                                    "max_length": max(120, int(words * 0.55)),
                                    "min_length": max(60, int(words * 0.3))
                                }
                            else:  # long
                                return {
                                    "max_length": max(180, int(words * 0.7)),
                                    "min_length": max(80, int(words * 0.4))
                                }

                        length_config = get_summary_lengths(summary_length, content_length)

                        # use this when calling summarizer
                        sum_result = summarizer.summarize(full_text, **length_config)

                        
                        if not sum_result.get('error'):
                            summary_text = sum_result['summary']
                        else:
                            summary_text = full_text[:600]
                    else:
                        summary_text = full_text
                    
                    # Clean summary too
                    summary_text = re.sub(r'https?://\S+', '', summary_text)
                    summary_text = re.sub(r'www\.\S+', '', summary_text)
                    summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                    
                    # Translate
                    trans_result = translator.translate(summary_text, target_lang, preprocess=False)
                    
                    st.session_state.translations[summary_key] = {
                        'text': trans_result['translated'],
                        'word_count': len(trans_result['translated'].split()),
                        'char_count': len(trans_result['translated'])
                    }
            
            # Display summary
            summary_data = st.session_state.translations[summary_key]
            st.markdown(f'<div class="translated-content">{summary_data["text"]}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Words", summary_data['word_count'])
            with col2:
                st.metric("Characters", summary_data['char_count'])
            
            st.divider()
            
            # Full article
            st.subheader("📖 Full Article")
            
            full_key = f"f_{idx}_{target_lang}"
            
            if full_key not in st.session_state.full_articles:
                if st.button("📖 Fetch & Translate Full Article", type="primary", use_container_width=True):
                    with st.spinner("⏳ Fetching article (10-30 sec)..."):
                        fetch_result = article_fetcher.fetch_article(article.get('url', ''))
                        
                        if fetch_result['success']:
                            with st.spinner("🔄 Translating (may take 30-60 sec)..."):
                                trans_result = translator.translate(fetch_result['content'], target_lang, preprocess=False)
                                
                                st.session_state.full_articles[full_key] = {
                                    'success': True,
                                    'content': trans_result['translated'],
                                    'stats': {
                                        'orig_words': fetch_result['word_count'],
                                        'trans_words': len(trans_result['translated'].split()),
                                        'chunks': trans_result.get('num_chunks', 1)
                                    }
                                }
                        else:
                            st.session_state.full_articles[full_key] = {
                                'success': False,
                                'error': fetch_result['error']
                            }
                    st.rerun()
            else:
                full_data = st.session_state.full_articles[full_key]
                
                if full_data['success']:
                    st.markdown(f'<div class="translated-content">{full_data["content"]}</div>', unsafe_allow_html=True)
                    
                    with st.expander("📊 Translation Stats"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Words", full_data['stats']['orig_words'])
                        with col2:
                            st.metric("Translated Words", full_data['stats']['trans_words'])
                        
                        if full_data['stats']['chunks'] > 1:
                            st.info(f"Translated in {full_data['stats']['chunks']} chunks")
                else:
                    st.error(f"❌ {full_data['error']}")
                    st.info("💡 Some websites block automated access. Try the original link above.")
        else:
            st.info("👈 Select an article to view translation")

else:
    st.info("👈 Click 'Fetch News' to get started!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🌍 Translation\n- 25+ languages\n- Context-aware\n- Long articles")
    with col2:
        st.markdown("### 📝 Summarization\n- 3 lengths\n- Key points\n- Multilingual")
    with col3:
        st.markdown("### 📰 News\n- 10 countries\n- 7 categories\n- Real-time")

st.divider()
st.caption("Powered by mBART-50, BART-CNN & NewsAPI")