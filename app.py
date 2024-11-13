from tracemalloc import stop
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from autoscraper import AutoScraper

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
sw = nltk.corpus.stopwords.words("english")

# Sidebar navigation
rad = st.sidebar.radio("Navigation", [
    "Home", 
    "Redmi Moonstone Silver (6GB 128GB)", 
    "Redmi 12 5G (256GB 8GB RAM Silver)", 
    "Samsung Galaxy Fold6 (256GB 12GB RAM Light Pink)", 
    "Realme Narzo 70x 5G (128GB 4GB RAM Ice Blue)"
])

# Home Page
if rad == "Home":
    st.title("SmartChoice")
    st.image("SmartChoicePic.png")
    st.text(" ")
    st.text("Make your choice of these top-level products:")
    st.text(" ")
    st.text("1. Redmi Moonstone Silver (6GB 128GB)")
    st.text("2. Redmi 12 5G (256GB 8GB RAM Silver)")
    st.text("3. Samsung Galaxy Fold6 (256GB 12GB RAM Light Pink)")
    st.text("4. Realme Narzo 70x 5G (128GB 4GB RAM Ice Blue)")

# Function to clean and transform the user input in raw format
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in sw and i not in string.punctuation]
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in y])

# Sentiment Analysis Prediction with CountVectorizer
count_vect = CountVectorizer(stop_words=sw, max_features=20)

def transform2(txt1):
    txt2 = count_vect.fit_transform(txt1)
    return txt2.toarray()

# Load and prepare the data for training
df2 = pd.read_csv("C:/Users/Bidisha/Documents/GitHub/SmartChoice/Sentiment Analysis.csv")
df2.columns = ["Text", "Label"]
x = transform2(df2["Text"])
y = df2["Label"]
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.1, random_state=0)
model2 = LogisticRegression()
model2.fit(x_train2, y_train2)

# Sentiment Analysis Function
def sentimentalAnalysis(result2):
    posCount = 0
    negCount = 0
    
    # Check if result2 is a dictionary, and access each review list
    for key, reviews in result2.items():
        for review in reviews:  # Iterate through each review under the key
            transformed_sent2 = transform_text(review)  # Clean the review text
            vector_sent2 = count_vect.transform([transformed_sent2])  # Transform to count vector
            prediction2 = model2.predict(vector_sent2)[0]  # Predict sentiment
            
            # Count positive and negative predictions
            if prediction2 == 1:
                posCount += 1
            else:
                negCount += 1
                
    return posCount, negCount

# Web scraping function for Amazon
def webScraping_amazon(url):
    amazon_url = "https://www.amazon.in/dp/B0D73TQLFZ?th=1"
    wanted_list1 = ["1,52,499"]
    wanted_list2 = ["Fabulous Device"]
    scraper = AutoScraper()
    
    # Build scraper for price
    result1 = scraper.build(amazon_url, wanted_list1)
    result1 = scraper.get_result_similar(url, group_by_alias=True)
    
    st.header("Amazon")
    if result1:
        first_key = next(iter(result1))
        first_value = result1[first_key][0] if result1[first_key] else "No result"
        st.success("Price: "+first_value)
    else:
        st.warning("Product not available!!!!")
    
    # Build scraper for review
    result2 = scraper.build(amazon_url, wanted_list2)
    results = scraper.get_result_similar(url, group_by_alias=True)
    return results

# Web scraping function for Reliance Digital
def webScraping_reliance_digital(url):
    reliance_url = "https://www.reliancedigital.in/samsung-galaxy-s23-ultra-5g-256-gb-12-gb-ram-phantom-black-mobile-phone/p/493665085"
    wanted_list1 = ["1,09,999.00"]
    wanted_list2 = ["Great all round package"]
    scraper = AutoScraper()
    
    # Build scraper for price
    result1 = scraper.build(reliance_url, wanted_list1)
    result1 = scraper.get_result_similar(url, group_by_alias=True)
    
    st.header("Reliance Digital")
    if result1:
        first_key = next(iter(result1))
        first_value = result1[first_key][0] if result1[first_key] else "No result"
        st.success("Price: "+first_value)
    else:
        st.warning("Product not available!!!!")
    
    # Build scraper for review
    result2 = scraper.build(reliance_url, wanted_list2)
    results = scraper.get_result_similar(url, group_by_alias=True)
    return results

# Updated Product sections to use web scraping and sentiment analysis
if rad == 'Redmi Moonstone Silver (6GB 128GB)':
    st.header("The Price and Review Analysis from Various Platforms!!")
    amazon_url="https://www.amazon.in/Redmi-Moonstone-Silver-6GB-128GB/dp/B0C9J9CZR6"
    result2 = webScraping_amazon(amazon_url)
    amazon_pos, amazon_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {amazon_pos}, Negative Reviews: {amazon_neg}")
    
    reliance_url="https://www.reliancedigital.in/redmi-12-5g-256-gb-8-gb-ram-silver-mobile-phone/p/493838734"
    result2 = webScraping_reliance_digital(reliance_url)
    reliance_pos, reliance_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {reliance_pos}, Negative Reviews: {reliance_neg}")

if rad == 'Redmi 12 5G (256GB 8GB RAM Silver)':
    st.header("The Price and Review Analysis from Various Platforms!!")
    amazon_url="https://www.amazon.in/Redmi-Moonstone-Silver-6GB-128GB/dp/B0C9J9CZR6"
    result2 = webScraping_amazon(amazon_url)
    amazon_pos, amazon_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {amazon_pos}, Negative Reviews: {amazon_neg}")
    
    reliance_url="https://www.reliancedigital.in/redmi-12-5g-256-gb-8-gb-ram-silver-mobile-phone/p/493838734"
    result2 = webScraping_reliance_digital(reliance_url)
    reliance_pos, reliance_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {reliance_pos}, Negative Reviews: {reliance_neg}")

if rad == 'Samsung Galaxy Fold6 (256GB 12GB RAM Light Pink)':
    st.header("The Price and Review Analysis from Various Platforms!!")
    amazon_url = "https://www.amazon.in/dp/B0D73TQLFZ"
    result2 = webScraping_amazon(amazon_url)
    amazon_pos, amazon_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {amazon_pos}, Negative Reviews: {amazon_neg}")
    
    reliance_url = "https://www.reliancedigital.in/samsung-galaxy-fold6-256-gb-12-gb-ram-light-pink-mobile-phone/p/494421980"
    result2 = webScraping_reliance_digital(reliance_url)
    reliance_pos, reliance_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {reliance_pos}, Negative Reviews: {reliance_neg}")

if rad == 'Realme Narzo 70x 5G (128GB 4GB RAM Ice Blue)':
    st.header("The Price and Review Analysis from Various Platforms!!")
    amazon_url="https://www.amazon.in/realme-Storage-Display-Dimensity-Charger/dp/B0CZS3B3PY"
    result2 = webScraping_amazon(amazon_url)
    amazon_pos, amazon_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {amazon_pos}, Negative Reviews: {amazon_neg}")
    
    reliance_url = "https://www.reliancedigital.in/realme-narzo-70x-5g-128-gb-4-gb-ram-ice-blue-mobile-phone/p/494422958"
    result2 = webScraping_reliance_digital(reliance_url)
    reliance_pos, reliance_neg = sentimentalAnalysis(result2)
    st.success(f"Positive Reviews: {reliance_pos}, Negative Reviews: {reliance_neg}")
