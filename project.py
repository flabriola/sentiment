import requests
from datetime import datetime, date
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# News API credentials
api_key = 'NEWS_API_KEY'
url = 'https://newsapi.org/v2/everything'
model = tf.keras.models.load_model('model.h5')            # ML model to predict sentiment
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>') # Tokenizer 
mode_str = "For sentiment based on topic(s) enter ANY key, to choose a timeframe enter T: "

def main():
    
    # Retrieve articles in initializing function
    tags, articles = init()
    tokenizer.fit_on_texts(articles)
    
    positive, negative, neutral = {'articles': [], 'count': 0}, {'articles': [], 'count': 0}, {'articles': [], 'count': 0}
    
    print(f"\n{spacer(mode_str)}\nAnalysing sentiment for 100 article on '{' '.join(tags)}'\n")
    
    for article in articles:
        sentiment = get_sentiment(article)
        
        if sentiment == 'positive':
            positive['articles'].append(article)
            positive['count'] += 1
        
        elif sentiment == 'negative':
            negative['articles'].append(article)
            negative['count'] += 1
            
        elif sentiment == 'neutral':
            neutral['articles'].append(article)
            neutral['count'] += 1
    
    print(f"\n{spacer(mode_str)}\n")
            
    print(f"Positive: {positive['count']}%, Negative: {negative['count']}%, Neutral: {neutral['count']}%\n")
    
def init():
    """
    Introduces user to program and makes parameters requests 
    """        
    
    # Intoduce program and request mode
   
    print(f"\nWelcome to SysSentient !\n\nA program that evaluates general sentiment on any topic(s), given a timeframe.\n\n{spacer(mode_str)}\n\n{mode_str}", end="")
   
    mode = input()
    
    print(f"\n{spacer(mode_str + mode)}\n")
    
    f = ""

    # Handle timeframe mode
    if mode.lower() == "t":
        f, t = request_timeframe(); print("\n")
        print("\n")
    
    # Request topics, get articles and handle error
    while True:
        print("Please enter desired topics in the form of tags. i.e 'bitcoin', 'paris olympics', 'climate change bill gates'\nDo not use any punctuation:\n")
        while True:
            tags = input("").strip().split()   # Receive tags and save as array of words
            if tags:
                break
        
        # Join words into query format
        q = ' OR '.join(tags) if len(tags) > 1 else tags[0]
        
        # For articles within time frame
        if f: 
            articles = fetch_articles(q, f, t)
            
        # No time frame
        else:
            articles = fetch_articles(q)
        
        if articles != 'error':
            break
    
    contents = []
    for article in articles:
        text = f"{article['title']} {article['description']}" #{article['content']}
        contents.append(text)
        
    return tags, contents


def spacer(s):
    # Return line from left to right (size of string or terminal)
    terminal_width = os.get_terminal_size().columns
    return f"{"-" * terminal_width if terminal_width < len(s) else "-" * len(s)}"


def request_timeframe():  
    
    # Request and return 'from' and 'to' dates as string tuple
    print(f"Enter date (YEAR-MONTH-DAY) or (TODAY) for 'to'")
    while True:
        try:
            from_d = input("From: ")
            f = datetime.strptime(from_d, '%Y-%m-%d')
            break
        except ValueError:
            pass
    
    while True:
        try:
            to_d = input("To: ").strip().lower()
            if to_d == 'today':
                break
            # Handle 'to' date being being from date
            if datetime.strptime(to_d, '%Y-%m-%d') <= f:
                print("'To' date must be after 'from' date")
                continue
            break
        except ValueError:
            pass
    
    return from_d, to_d
    
    
def fetch_articles(q, page_size=100, f=None, t=None):
    params = {
        'q': q,
        'pageSize': page_size,
        'apiKey': api_key
    }
    if f:
        params['from'] = f
    
    if t != 'today' and t is not None:
        params['to'] = t
        
    response = requests.get(url, params=params)
    
    return (
        response.json()['status'] if response.json()['status'] == 'error' else
        response.json()['articles']
        )
        

def get_sentiment(article):
    
    # Tokenize and pad sequences 
    sequences = tokenizer.texts_to_sequences([article])
    
    text_padded = pad_sequences(sequences, padding='post', maxlen=100)
    
    # Make prediction using ML model and format response
    sentiment = model.predict(text_padded)
    
    # Get the predicted class (0 = negative, 1 = neutral, 2 = positive)
    predicted_class = np.argmax(sentiment, axis=1)[0]
    
    # Map the predicted class to the corresponding sentiment label
    sentiment_label = ['negative', 'neutral', 'positive'][predicted_class]
    
    return sentiment_label


if __name__ == "__main__":
    main()