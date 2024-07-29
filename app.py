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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from autoscraper import AutoScraper

nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")

rad=st.sidebar.radio("Navigation",["Home","Apple iPhone 15 Pro","Apple IPhone pro max 256","Samsung Galaxy Z Fold6 5G AI Smartphone","Samsung Galaxy S24 Ultra 5G AI Smartphone"])

#Home Page
if rad=="Home":
    st.title("SmartChoice")
    st.image("SmartChoicePic.png")
    st.text(" ")
    st.text("Make your choice of this top level products->")
    st.text(" ")
    st.text("1. Apple iPhone 15 Pro")
    st.text("2. Apple IPhone pro max 256")
    st.text("3. Samsung Galaxy Z Fold6 5G AI Smartphone")
    st.text("4. Samsung Galaxy S24 Ultra 5G AI Smartphone")



#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


#Sentiment Analysis Prediction 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt1):
    txt2=tfidf2.fit_transform(txt1)
    return txt2.toarray()

df2=pd.read_csv("D:/projects/Bidisa/SmartChoice/Sentiment Analysis.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
y=df2["Label"]
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
model2=LogisticRegression()
model2.fit(x_train2,y_train2)

#Sentiment Analysis Page
def sentimentalAnalysis(result2):
    for i in result2:
        transformed_sent2=transform_text(i)
        vector_sent2=tfidf2.transform([transformed_sent2])
        prediction2=model2.predict(vector_sent2)[0]
        break
 
    return prediction2


#web scraping
def webScraping_amazon(url):
    amazon_url = "https://www.amazon.in/Apple-iPhone-Pro-Max-256/dp/B0CHWV2WYK"
    wanted_list1=["1,51,700"]
    wanted_list2=["Excellent device at any expects."]
    scraper=AutoScraper()
    result1=scraper.build(amazon_url,wanted_list1)
    st.header("Amazon")
    if result1!= []:
       st.success(result1)
    else:
        st.warning("Product not available!!!!")
    result2=scraper.build(amazon_url,wanted_list2)
    results=scraper.get_result_similar(url,group_by_alias=True)
    return results

#webScraping
def webScraping_flipkart(url):
    amazon_url = "https://www.flipkart.com/apple-iphone-15-pro-max-blue-titanium-256-gb/p/itm4a0093df4a3d7"
    wanted_list1=["1,39,990"]
    wanted_list2=["Good design, thinner bezzel, good performance, good battery life, outstanding 5x optical zoom camera"]
    scraper=AutoScraper()
    result1=scraper.build(amazon_url,wanted_list1)
    st.header("Flipkart")
    if result1!= []:
       st.success(result1)
    else:
        st.warning("Product not available!!!!")
    result2=scraper.build(amazon_url,wanted_list2)
    results=scraper.get_result_similar(url,group_by_alias=True)
    return results



#products

if rad=='Apple IPhone pro max 256':
    st.header("The Price and review analysis from various plateforms!!")
    amazon_url = "https://www.amazon.in/dp/B0CHWV2WYK"
    result2=webScraping_amazon(amazon_url)
    amazon_prediction=sentimentalAnalysis(result2)
    if amazon_prediction==0:
        st.warning("Negetive Review!!")
    elif amazon_prediction==1:
        st.success("Positive Review!!")
    flipkart_url="https://www.flipkart.com/apple-iphone-15-pro-max-blue-titanium-256-gb/p/itm4a0093df4a3d7"
    result2=webScraping_flipkart(flipkart_url)
    flipkart_prediction=sentimentalAnalysis(result2)
    if flipkart_prediction==0:
        st.warning("Negetive Review!!")
    elif flipkart_prediction==1:
        st.success("Positive Review!!")

if rad=='Apple iPhone 15 Pro':
    st.header("The Price and review analysis from various plateforms!!")
    amazon_url = "https://www.amazon.in/dp/B0CHX7J4TL"
    result2=webScraping_amazon(amazon_url)
    amazon_prediction=sentimentalAnalysis(result2)
    if amazon_prediction==0:
        st.warning("Negetive Review!!")
    elif amazon_prediction==1:
        st.success("Positive Review!!")
    flipkart_url="https://www.flipkart.com/apple-iphone-15-pro-black-titanium-128-gb/p/itm96f61fdd7e604"
    result2=webScraping_flipkart(flipkart_url)
    flipkart_prediction=sentimentalAnalysis(result2)
    if flipkart_prediction==0:
        st.warning("Negetive Review!!")
    elif flipkart_prediction==1:
        st.success("Positive Review!!")

if rad=='Samsung Galaxy Z Fold6 5G AI Smartphone':
    st.header("The Price and review analysis from various plateforms!!")
    amazon_url = "https://www.amazon.in/dp/B0D73TQLFZ"
    result2=webScraping_amazon(amazon_url)
    amazon_prediction=sentimentalAnalysis(result2)
    if amazon_prediction==0:
        st.warning("Negetive Review!!")
    elif amazon_prediction==1:
        st.success("Positive Review!!")
    flipkart_url="https://www.flipkart.com/samsung-galaxy-z-fold6-5g-navy-512-gb/p/itm4cad29eca0a90"
    result2=webScraping_flipkart(flipkart_url)
    flipkart_prediction=sentimentalAnalysis(result2)
    if flipkart_prediction==0:
        st.warning("Negetive Review!!")
    elif flipkart_prediction==1:
        st.success("Positive Review!!")

if rad=='Samsung Galaxy S24 Ultra 5G AI Smartphone':
    st.header("The Price and review analysis from various plateforms!!")
    amazon_url = "https://www.amazon.in/dp/B0CS6JW9YQ"
    result2=webScraping_amazon(amazon_url)
    amazon_prediction=sentimentalAnalysis(result2)
    if amazon_prediction==0:
        st.warning("Negetive Review!!")
    elif amazon_prediction==1:
        st.success("Positive Review!!")
    flipkart_url="https://www.flipkart.com/samsung-galaxy-s24-ultra-5g-titanium-gray-512-gb/p/itm463827d6eb2be"
    result2=webScraping_flipkart(flipkart_url)
    flipkart_prediction=sentimentalAnalysis(result2)
    if flipkart_prediction==0:
        st.warning("Negetive Review!!")
    elif flipkart_prediction==1:
        st.success("Positive Review!!")
