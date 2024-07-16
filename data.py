import pandas as pd
import json

"""
Data Preparation
Parsing all datasets to include only a body of text and sentiment score (-1, 0, 1) 
"""

# Main DataFrame
df = pd.DataFrame(columns=['text', 'sentiment'])

# Twitter Dataset training.csv
df1 = pd.read_csv('Twitter_Data.csv')

df1.dropna(subset=['clean_text', 'category'], inplace=True)
df1 = df1[df1['clean_text'].str.strip() != '']
df1 = df1[df1['category'].isin([-1, 0, 1])]
df1['category'] = df1['category'].astype(int)

# News Dataset rating.csv
df2 = pd.read_csv('rating.csv')
df2 = df2.drop(columns=['article_id', 'source_id', 'source_name', 'author', 'title', 'description', 'url', 'url_to_image', 'published_at', 'content', 'category'])
df2 = df2[df2['article'].str.strip() != '']
df2['title_sentiment'] = df2['title_sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
df2 = df2[df2['title_sentiment'].isin([-1, 0, 1])]

# News Dataset train.jsonl
with open('train.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data = []
    for line in lines:
        entry = json.loads(line.strip())
        if entry['polarity'] in [-1, 0, 1] and entry['sentence'].strip():
            data.append({'text': entry['sentence'], 'sentiment': entry['polarity']})

# Add all articles, tweets and senteces, as well as their sentiment scores to main df
df['text'] = pd.concat([df1['clean_text'], df2['article']], ignore_index=True)
df['sentiment'] = pd.concat([df1['category'], df2['title_sentiment']], ignore_index=True)
df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

output_csv_path = 'training.csv'
df.to_csv(output_csv_path, index=False)

print(f"Prepared data saved to: {output_csv_path}")