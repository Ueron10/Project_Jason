import streamlit as st
import numpy as np
import pickle
import re
import os
import nltk
import time
import tensorflow as tf
from bs4 import BeautifulSoup
from transformers import TFBertForSequenceClassification, BertTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Initialize preprocessing tools - Optimized for BERT
minimal_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'up', 'down', 'out', 'over', 'under',
                    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                    'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                    'won', "won't", 'wouldn', "wouldn't"}
stop_words = minimal_stopwords

@st.cache_resource
def load_artifacts():
    model = TFBertForSequenceClassification.from_pretrained(os.path.join(models_dir, "bert_model"))
    with open(os.path.join(outputs_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)
    # Ensure correct mapping: index 0 = negative, index 1 = positive
    # If label_classes is ['negative', 'positive'], use as is
    # If reversed, we need to handle it in prediction
    return model, tokenizer, label_classes

# Load artifacts
model, tokenizer, label_classes = load_artifacts()

# Verify label mapping (debug info)
# st.sidebar.write("Label mapping:", {i: label_classes[i] for i in range(len(label_classes))})

def expand_contractions(text):
    """Expand common contractions for better BERT understanding"""
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "can't": "cannot",
        "couldn't": "could not", "shouldn't": "should not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "i'm": "i am", "you're": "you are", "he's": "he is",
        "she's": "she is", "it's": "it is", "we're": "we are",
        "they're": "they are", "i've": "i have", "you've": "you have",
        "we've": "we have", "they've": "they have", "i'll": "i will",
        "you'll": "you will", "he'll": "he will", "she'll": "she will",
        "we'll": "we will", "they'll": "they will", "i'd": "i would",
        "you'd": "you would", "he'd": "he would", "she'd": "she would",
        "we'd": "we would", "they'd": "they would"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text):
    """Optimized preprocessing for BERT - keeps more context"""
    if not text:
        return ""
    
    # Expand contractions first
    text = expand_contractions(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Keep letters, numbers, and basic punctuation (BERT handles punctuation well)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    if not cleaned:
        return None
    # BERT Large tokenization
    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='tf'
    )
    # BERT Large prediction
    logits = model.predict(
        {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']},
        verbose=0
    )
    probs = tf.nn.softmax(logits.logits, axis=-1).numpy()[0]
    labels = [str(label).lower() for label in label_classes]
    probabilities = {labels[i]: float(probs[i]) for i in range(len(labels))}
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx])
    return {
        'label': pred_label,
        'confidence': confidence,
        'probabilities': probabilities,
        'cleaned': cleaned
    }

def scrape_imdb_selenium(url, max_reviews=None):
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service as ChromeService
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            try:
                driver = webdriver.Chrome(options=chrome_options)
            except:
                return {"error": f"Chrome driver not found. Please install Chrome browser and chromedriver. Details: {str(e)}", "reviews": [], "count": 0}
        
        driver.get(url)
        time.sleep(5)
        
        reviews_data = []
        last_count = 0
        no_change_count = 0
        max_no_change = 5
        target_count = max_reviews if max_reviews else 1000
        
        # Scroll dan load more sampai dapat target atau tidak ada lagi
        scroll_attempts = 0
        max_scroll_attempts = 50
        
        while len(reviews_data) < target_count and scroll_attempts < max_scroll_attempts:
            scroll_attempts += 1
            
            # Method 1: Scroll ke bawah window
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Method 2: Scroll ke semua review elements yang sudah ada
            try:
                review_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="review-card"], .ipc-list-card__content')
                if review_elements:
                    # Scroll ke element terakhir
                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", review_elements[-1])
                    time.sleep(1)
                    
                    # Scroll ke beberapa element di tengah untuk trigger lazy load
                    mid_index = len(review_elements) // 2
                    if mid_index > 0 and mid_index < len(review_elements):
                        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", review_elements[mid_index])
                        time.sleep(1)
            except:
                pass
            
            # Method 3: Klik Load More dengan explicit wait
            try:
                load_more = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, '//button[contains(text(),"Load More")]'))
                )
                driver.execute_script("arguments[0].click();", load_more)
                print("Clicked Load More button")
                time.sleep(4)  # Wait longer after click
            except:
                pass
            
            # Method 4: Try to find and click any expandable buttons + See all
            try:
                buttons = driver.find_elements(By.XPATH, '//button[contains(@class, "button") or contains(@class, "btn") or contains(@class, "ipc-btn")]')
                for btn in buttons:
                    try:
                        btn_text = btn.text.lower()
                        if any(keyword in btn_text for keyword in ['load', 'more', 'expand', 'see all', 'seeall', 'view all', 'viewall']):
                            if btn.is_displayed() and btn.is_enabled():
                                driver.execute_script("arguments[0].click();", btn)
                                print(f"Clicked button: {btn_text}")
                                
                                # Tunggu loading selesai - cek berbagai indikator
                                loading_wait = 0
                                max_loading_wait = 30  # Max 30 detik tunggu
                                while loading_wait < max_loading_wait:
                                    time.sleep(1)
                                    loading_wait += 1
                                    
                                    # Cek apakah masih loading
                                    loading_indicators = driver.find_elements(By.XPATH, 
                                        '//div[contains(@class, "loading") or contains(@class, "spinner") or contains(@class, "progress")] | //span[contains(text(),"Loading")]')
                                    
                                    if not loading_indicators:
                                        # Cek apakah content bertambah
                                        html_check = driver.page_source
                                        soup_check = BeautifulSoup(html_check, "html.parser")
                                        current_reviews = soup_check.find_all('div', class_="ipc-list-card__content")
                                        if len(current_reviews) > last_count:
                                            print(f"Loading done. Reviews increased from {last_count} to {len(current_reviews)}")
                                            # Extra wait 10 detik untuk pastikan semua terload
                                            print("Waiting 10 seconds for full render...")
                                            time.sleep(10)
                                            break
                                    
                                    if loading_wait % 5 == 0:
                                        print(f"Still waiting for loading... ({loading_wait}s)")
                                
                                break
                    except Exception as e:
                        print(f"Button click error: {str(e)[:50]}")
                        continue
            except Exception as e:
                print(f"Button scan error: {str(e)[:50]}")
            
            # Method 5: Check for pagination / Next button
            try:
                next_buttons = driver.find_elements(By.XPATH, 
                    '//a[contains(text(),"Next")] | //button[contains(text(),"Next")] | //a[@aria-label="Next"]')
                for next_btn in next_buttons:
                    if next_btn.is_displayed() and next_btn.is_enabled():
                        driver.execute_script("arguments[0].click();", next_btn)
                        print("Clicked Next page button")
                        time.sleep(4)
                        break
            except:
                pass
            
            # Method 6: Check total review count on page
            try:
                count_elements = driver.find_elements(By.XPATH, 
                    '//span[contains(text(),"reviews") or contains(@class, "count")] | //div[contains(text(),"showing")]')
                for elem in count_elements:
                    print(f"Page shows: {elem.text}")
            except:
                pass
            
            # Ambil review dengan multiple selectors
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            
            # Coba berbagai selector yang mungkin
            reviews = soup.find_all('div', class_="ipc-list-card__content")
            if not reviews:
                reviews = soup.find_all('div', class_="review-container")
            if not reviews:
                reviews = soup.find_all('div', attrs={"data-testid": "review-card"})
            if not reviews:
                # Selector generik untuk article/review
                reviews = soup.find_all('article') or soup.find_all('div', class_=lambda x: x and 'review' in x.lower())
            
            current_count = len(reviews)
            print(f"Found {current_count} review containers...")
            
            # Parse reviews yang belum ada
            new_reviews_found = 0
            for review in reviews:
                # Cari title dengan berbagai kemungkinan
                title_elem = review.find('h3') or review.find('a', class_=lambda x: x and 'title' in str(x).lower())
                title = title_elem.text.strip() if title_elem else ""
                
                # Cari content dengan berbagai kemungkinan  
                content_elem = (review.find('div', class_="ipc-html-content-inner-div") or 
                             review.find('div', class_=lambda x: x and 'content' in str(x).lower()) or
                             review.find('span', class_=lambda x: x and 'content' in str(x).lower()) or
                             review.find('p'))
                content = content_elem.get_text(strip=True) if content_elem else ""
                
                # Cek duplicate dan filter
                if content and len(content) > 20:  # Minimal 20 chars
                    is_duplicate = any(r['content'] == content for r in reviews_data)
                    if not is_duplicate:
                        reviews_data.append({"title": title, "content": content})
                        new_reviews_found += 1
                        
                        # Stop kalau sudah cukup
                        if max_reviews and len(reviews_data) >= max_reviews:
                            break
            
            print(f"Added {new_reviews_found} new reviews. Total: {len(reviews_data)}")
            
            # Cek apakah masih ada review baru
            if new_reviews_found == 0:
                no_change_count += 1
                if no_change_count >= max_no_change:
                    print("No more new reviews found. Stopping.")
                    break
            else:
                no_change_count = 0
            
            # Extra wait untuk lazy load
            time.sleep(1)
        
        # Final check: ambil lagi kalau masih kurang
        if len(reviews_data) < (max_reviews or 1):
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            
            # Last resort: cari semua text yang panjang
            all_long_texts = soup.find_all(text=lambda text: text and len(str(text)) > 100)
            print(f"Debug: Found {len(all_long_texts)} long text elements")
        
        # Limit sesuai request
        if max_reviews:
            reviews_data = reviews_data[:max_reviews]
        
        if not reviews_data:
            return {"error": "No reviews found", "reviews": [], "count": 0}
        
        return {"reviews": reviews_data, "count": len(reviews_data)}
        
    except Exception as e:
        return {"error": str(e), "reviews": [], "count": 0}
    finally:
        if driver:
            try:
                time.sleep(10)
                driver.quit()
            except:
                pass

st.title("Sentiment Analysis")
st.markdown("Analyze movie review sentiment using CNN")

tab1, tab2 = st.tabs(["Text Input", "IMDB Reviews"])

with tab1:
    st.header("Enter Text")
    user_text = st.text_area("Input your review:", height=200, 
                             placeholder="Example: This movie was absolutely fantastic!")
    
    if st.button("Analyze", key="text_btn"):
        if not user_text.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner("Analyzing..."):
                result = predict_sentiment(user_text)
                if result:
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment = result['label'].upper()
                        color = "green" if sentiment == "POSITIVE" else "red"
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {color}20; 
                                    border-left: 5px solid {color};">
                            <h2 style="margin: 0; color: {color};">{sentiment}</h2>
                            <p style="margin: 5px 0 0 0; font-size: 18px;">
                                Confidence: <b>{result['confidence']:.2%}</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Probabilities:**")
                        for label, prob in result['probabilities'].items():
                            st.progress(prob, text=f"{label}: {prob:.2%}")
                    
                    with st.expander("View preprocessed text"):
                        st.text(result['cleaned'])

with tab2:
    st.header("IMDB Reviews (Selenium)")
    st.warning("This mode opens Chrome browser to scrape IMDB JavaScript-loaded reviews. Slower but works for IMDB.")
    
    imdb_url = st.text_input("IMDB Reviews URL:", 
                             placeholder="https://www.imdb.com/title/tt16311594/reviews/",
                             key="imdb_input")
    
    scrape_option = st.radio("Scrape mode:", ["Specific count", "All reviews"])
    
    if scrape_option == "Specific count":
        max_reviews = st.number_input("Number of reviews to scrape:", min_value=1, max_value=100, value=5, step=1)
    else:
        max_reviews = None
        st.info("Will scrape all available reviews until no more 'Load More' button.")
    
    if st.button("Scrape IMDB & Analyze", key="imdb_btn"):
        if not imdb_url.strip():
            st.warning("Please enter IMDB URL!")
        elif "imdb.com" not in imdb_url:
            st.warning("Please enter valid IMDB URL!")
        else:
            with st.spinner("Opening Chrome and scraping... (this may take 30-60 seconds or more for all reviews)"):
                result = scrape_imdb_selenium(imdb_url, max_reviews)
                if result.get("error"):
                    st.error(f"Failed: {result['error']}")
                else:
                    reviews = result['reviews']
                    actual_count = result['count']
                    st.success(f"Scraped {actual_count} reviews from IMDB")
                    
                    st.markdown("### Individual Review Predictions")
                    
                    positive_count = 0
                    negative_count = 0
                    
                    for i, review in enumerate(reviews, 1):
                        review_text = f"{review['title']}: {review['content']}"
                        prediction = predict_sentiment(review_text)
                        
                        if prediction:
                            sentiment = prediction['label'].upper()
                            if sentiment == "POSITIVE":
                                positive_count += 1
                                color = "green"
                            else:
                                negative_count += 1
                                color = "red"
                            
                            with st.expander(f"Review {i}: {sentiment} ({prediction['confidence']:.0%})"):
                                st.markdown(f"**Title:** {review['title']}")
                                st.markdown(f"**Sentiment:** <span style='color:{color};font-weight:bold;'>{sentiment}</span> ({prediction['confidence']:.2%})", unsafe_allow_html=True)
                                st.markdown(f"**Content:** {review['content'][:300]}...")
                    
                    st.markdown("---")
                    st.markdown("### Overall Summary")
                    total = positive_count + negative_count
                    if total > 0:
                        pos_pct = (positive_count / total) * 100
                        neg_pct = (negative_count / total) * 100
                        
                        overall = "POSITIVE" if positive_count >= negative_count else "NEGATIVE"
                        overall_color = "green" if overall == "POSITIVE" else "red"
                        
                        st.markdown(f"**Overall Sentiment:** <span style='color:{overall_color};font-size:24px;font-weight:bold;'>{overall}</span>", unsafe_allow_html=True)
                        st.markdown(f"- Positive: {positive_count} ({pos_pct:.1f}%)")
                        st.markdown(f"- Negative: {negative_count} ({neg_pct:.1f}%)")

st.markdown("---")
st.caption("CNN Model | Binary: Positive/Negative")
