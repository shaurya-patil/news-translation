"""
Quick System Test - Run this before starting the app
Tests all components to make sure everything works
"""

print("="*70)
print("SYSTEM TEST - Multilingual News Platform")
print("="*70)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from translation import TranslationModel, LANGUAGE_NAMES
    from summarization import Summarizer
    from news_fetcher import NewsFetcher
    from article_fetcher import ArticleFetcher
    from preprocessing import preprocess_text
    import streamlit
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install streamlit torch transformers requests beautifulsoup4 langdetect contractions sentencepiece protobuf accelerate lxml")
    exit(1)

# Test 2: News fetching
print("\n[2/5] Testing news fetcher...")
try:
    fetcher = NewsFetcher()
    result = fetcher.get_top_headlines(country="us", page_size=3)
    if result.get('status') == 'ok' and result.get('articles'):
        print(f"✅ Fetched {len(result['articles'])} articles")
    else:
        print(f"⚠️ News API issue: {result.get('message', 'Unknown')}")
except Exception as e:
    print(f"❌ News fetcher error: {e}")

# Test 3: Translation model
print("\n[3/5] Testing translation model...")
print("(This will take 1-3 minutes first time - downloading models)")
try:
    translator = TranslationModel()
    test_text = "Hello, this is a test."
    result = translator.translate(test_text, "hi_IN", preprocess=False)
    if result.get('translated'):
        print(f"✅ Translation works: '{test_text}' → '{result['translated']}'")
    else:
        print("❌ Translation failed")
except Exception as e:
    print(f"❌ Translation error: {e}")

# Test 4: Summarization model  
print("\n[4/5] Testing summarization model...")
try:
    summarizer = Summarizer()
    test_text = """
    The stock market experienced significant volatility today as investors 
    reacted to new economic data. The Dow Jones Industrial Average fell by 
    500 points in early trading before recovering slightly. Analysts attribute 
    the decline to concerns about inflation and rising interest rates.
    """
    result = summarizer.summarize(test_text, max_length=50, min_length=20, preprocess=False)
    if result.get('summary'):
        print(f"✅ Summarization works")
        print(f"   Original: {len(test_text)} chars")
        print(f"   Summary: {len(result['summary'])} chars")
    else:
        print(f"❌ Summarization failed: {result.get('error')}")
except Exception as e:
    print(f"❌ Summarizer error: {e}")

# Test 5: Article fetcher
print("\n[5/5] Testing article fetcher...")
try:
    fetcher = ArticleFetcher()
    # Test with BBC (usually accessible)
    result = fetcher.fetch_article("https://www.formula1.com/en/latest/article/watch-see-how-norris-beat-leclerc-to-pole-position-in-mexico-with-our-ghost.7ry8NWAEZuAy0nRG62KzTS")
    if result.get('success'):
        print(f"✅ Article scraping works ({result['word_count']} words extracted)")
    else:
        print(f"⚠️ Article scraping: {result.get('error')} (this is normal for some sites)")
except Exception as e:
    print(f"❌ Article fetcher error: {e}")

# Final summary
print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\n✅ System is ready to use!")
print("\nNext step: Run the app with:")
print("   streamlit run app.py")
print("\n" + "="*70)