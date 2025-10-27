"""
Main Streamlit Application
News Translator with Preprocessing, Translation, and Summarization
"""

import streamlit as st

# Import custom modules
from preprocessing import preprocess_text, detect_language, get_text_statistics
from translation import TranslationModel, LANGUAGE_NAMES
from summarization import Summarizer
from news_fetcher import NewsFetcher, format_article, filter_articles, get_article_statistics

# Page configuration
st.set_page_config(
    page_title="News Translator & Summarizer",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-box {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌐 News Translator & Summarizer</div>', unsafe_allow_html=True)
st.markdown("*Fetch news, preprocess text, translate to any language, and generate summaries*")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Module selection
    st.subheader("📌 Active Modules")
    show_preprocessing = st.checkbox("Text Preprocessing", value=True)
    show_translation = st.checkbox("Translation", value=True)
    show_summarization = st.checkbox("Summarization", value=True)
    
    st.divider()
    
    # Translation settings
    if show_translation:
        st.subheader("🌍 Translation Settings")
        target_lang = st.selectbox(
            "Translate to:",
            options=list(LANGUAGE_NAMES.keys()),
            format_func=lambda x: LANGUAGE_NAMES[x],
            index=0
        )
    
    # Summarization settings
    if show_summarization:
        st.subheader("📝 Summarization Settings")
        summary_length = st.select_slider(
            "Summary length:",
            options=["short", "medium", "long"],
            value="medium"
        )
        
        translate_summary = st.checkbox("Translate summary", value=False)
    
    st.divider()
    
    # News fetching settings
    st.subheader("📰 News Settings")
    
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
    
    search_query = st.text_input("🔍 Search keyword (optional):")
    num_articles = st.slider("Number of articles:", 5, 20, 10)
    
    st.divider()
    
    fetch_button = st.button("🔄 Fetch News", type="primary", use_container_width=True)

# Initialize models (cached in session state)
@st.cache_resource
def load_models():
    """Load all required models"""
    translator = TranslationModel() if show_translation else None
    summarizer = Summarizer() if show_summarization else None
    return translator, summarizer

# Load models
with st.spinner("🔄 Loading models..."):
    translator, summarizer = load_models()
    st.success("✅ Models loaded successfully!")

# Initialize news fetcher
news_fetcher = NewsFetcher()

# Main content area
if fetch_button:
    st.session_state.fetch_news = True

if st.session_state.get('fetch_news', False):
    # Fetch news
    with st.spinner("📡 Fetching news..."):
        news_data = news_fetcher.get_top_headlines(
            country=country,
            category=category if category != "all" else None,
            query=search_query if search_query else None,
            page_size=num_articles
        )
    
    if news_data.get('status') == 'ok':
        articles = news_data.get('articles', [])
        
        # Filter articles
        filtered_articles = filter_articles(articles, min_length=50)
        
        # Display statistics
        stats = get_article_statistics(filtered_articles)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📰 Total Articles", stats['total'])
        with col2:
            st.metric("📚 Sources", stats['source_count'])
        with col3:
            st.metric("📏 Avg Length", f"{stats['avg_length']:.0f} chars")
        with col4:
            st.metric("🔤 Language", "English")
        
        st.divider()
        
        # Display articles
        for idx, article in enumerate(filtered_articles):
            formatted = format_article(article)
            
            with st.container():
                # Article header
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.subheader(f"{idx + 1}. {formatted['title']}")
                    st.caption(f"📅 {formatted['published_at']} | 📰 {formatted['source']}")
                
                with col2:
                    if formatted['image_url']:
                        st.image(formatted['image_url'], width=150)
                
                # Tabs for different views
                tabs = []
                tab_names = ["📰 Original"]
                
                if show_preprocessing:
                    tab_names.append("🔧 Preprocessed")
                if show_translation:
                    tab_names.append("🌍 Translation")
                if show_summarization:
                    tab_names.append("📝 Summary")
                
                tabs = st.tabs(tab_names)
                tab_idx = 0
                
                # Original content
                with tabs[tab_idx]:
                    st.write(formatted['full_text'])
                    
                    # Text statistics
                    if show_preprocessing:
                        with st.expander("📊 Text Statistics"):
                            text_stats = get_text_statistics(formatted['full_text'])
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Words", text_stats['word_count'])
                            with col2:
                                st.metric("Sentences", text_stats['sentence_count'])
                            with col3:
                                st.metric("Avg Word Length", f"{text_stats['avg_word_length']:.1f}")
                    
                    if formatted['url']:
                        st.markdown(f"[🔗 Read full article]({formatted['url']})")
                tab_idx += 1
                
                # Preprocessed content
                if show_preprocessing:
                    with tabs[tab_idx]:
                        preprocessed = preprocess_text(formatted['full_text'])
                        detected_lang = detect_language(preprocessed)
                        
                        st.info(f"**Detected Language:** {detected_lang}")
                        st.write(preprocessed)
                        
                        st.caption("✨ Applied: Contraction expansion, URL removal, whitespace normalization")
                    tab_idx += 1
                
                # Translation
                if show_translation:
                    with tabs[tab_idx]:
                        if st.button(f"🌐 Translate to {LANGUAGE_NAMES[target_lang]}", key=f"translate_{idx}"):
                            with st.spinner("Translating..."):
                                result = translator.translate(formatted['full_text'], target_lang)
                                st.session_state[f'translation_{idx}'] = result
                        
                        if f'translation_{idx}' in st.session_state:
                            result = st.session_state[f'translation_{idx}']
                            st.success(f"✅ Translated from {result['source_lang']} to {result['target_lang']}")
                            st.write(result['translated'])
                            
                            if result.get('num_chunks', 1) > 1:
                                st.caption(f"ℹ️ Translated in {result['num_chunks']} chunks")
                    tab_idx += 1
                
                # Summarization
                if show_summarization:
                    with tabs[tab_idx]:
                        length_map = {"short": (50, 20), "medium": (130, 40), "long": (200, 80)}
                        max_len, min_len = length_map[summary_length]
                        
                        if st.button(f"📝 Generate {summary_length.title()} Summary", key=f"summarize_{idx}"):
                            with st.spinner("Generating summary..."):
                                result = summarizer.summarize(
                                    formatted['full_text'],
                                    max_length=max_len,
                                    min_length=min_len
                                )
                                
                                if translate_summary and show_translation:
                                    with st.spinner("Translating summary..."):
                                        trans_result = translator.translate(result['summary'], target_lang)
                                        result['translated_summary'] = trans_result['translated']
                                
                                st.session_state[f'summary_{idx}'] = result
                        
                        if f'summary_{idx}' in st.session_state:
                            result = st.session_state[f'summary_{idx}']
                            
                            if result.get('error'):
                                st.error(f"❌ Error: {result['error']}")
                            else:
                                st.success(f"✅ Summary generated (Compression: {result['compression_ratio']:.1%})")
                                
                                st.subheader("English Summary")
                                st.write(result['summary'])
                                
                                if result.get('translated_summary'):
                                    st.subheader(f"{LANGUAGE_NAMES[target_lang]} Summary")
                                    st.write(result['translated_summary'])
                    tab_idx += 1
                
                st.divider()
    
    else:
        st.error(f"❌ Error fetching news: {news_data.get('message', 'Unknown error')}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Modules:</strong> Preprocessing | Translation | Summarization | News API</p>
    <p>Powered by mBART-50, BART-CNN & NewsAPI</p>
</div>
""", unsafe_allow_html=True)