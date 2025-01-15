#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud


# In[11]:


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')


# # Pre-Processing Data

# In[7]:


tweets_data = pd.read_csv('tweets_data.csv')

print(tweets_data.info())


# In[9]:


import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('indonesian'))  

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'@\w+', '', text)    
    text = re.sub(r'#\w+', '', text)    
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  
    text = ' '.join([word for word in text.split() if word not in stop_words])  
    return text

tweets_data['cleaned_content'] = tweets_data['Content'].apply(clean_text)
print(tweets_data[['Content', 'cleaned_content']].head())


# # Sentimen Analysis

# In[13]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    score = analyzer.polarity_scores(text)
    return score['compound'] 

tweets_data['sentiment_score'] = tweets_data['cleaned_content'].apply(analyze_sentiment_vader)

tweets_data['sentiment_label'] = tweets_data['sentiment_score'].apply(
    lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
)

print(tweets_data[['cleaned_content', 'sentiment_score', 'sentiment_label']].head())


# In[25]:


#Sentiment Distribution
sns.countplot(data=tweets_data, x='sentiment_label', palette='coolwarm')
plt.title('Sentiment Distribution')
plt.show()


# In[26]:


positive_text = ' '.join(tweets_data[tweets_data['sentiment_label'] == 'positive']['cleaned_content'])
negative_text = ' '.join(tweets_data[tweets_data['sentiment_label'] == 'negative']['cleaned_content'])

# WordCloud for positif sentiment
positive_wc = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(positive_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Sentiment Positive')
plt.show()

# WordCloud for negatif sentiment
negative_wc = WordCloud(width=800, height=400, background_color='black').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(negative_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Sentiment Negative')
plt.show()


# In[27]:


#Time Based Sentiment Trends
tweets_data['Date'] = pd.to_datetime(tweets_data['Date'])

sentiment_trends = tweets_data.groupby(pd.Grouper(key='Date', freq='T'))['sentiment_score'].mean()

plt.figure(figsize=(12, 6))
sentiment_trends.plot()
plt.title('Sentiment Trends Per Minute')
plt.xlabel('Time (Per Minute)')
plt.ylabel('Average Sentiment Score')
plt.show()


# In[31]:


#Correlation of Sentiment with Interaction
correlation = tweets_data[['Likes', 'Retweets', 'Replies', 'sentiment_score']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation between Sentiment and Engagement')
plt.show()


# In[30]:


#Frequently Discussed Topics
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(tweets_data['cleaned_content'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(doc_term_matrix)

for i, topic in enumerate(lda.components_):
    print(f"Topik {i}:")
    print([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]])
    print("\n")


# In[33]:


#Interaction Distribution
interactions = ['Likes', 'Retweets', 'Replies']
tweets_data[interactions].sum().plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title('Distribution of Total Interactions')
plt.ylabel('Total')
plt.xticks(rotation=0)
plt.show()


# In[34]:


#Top tweets
top_interaction = tweets_data.sort_values(by='Likes', ascending=False).head(5)
print("Top 5 Tweets with the Highest Likes:")
print(top_interaction[['Date', 'Content', 'Likes']])


# In[36]:


#Words Frequency
from collections import Counter
all_words = ' '.join(tweets_data['cleaned_content']).split()
word_freq = Counter(all_words)

most_common_words = word_freq.most_common(20)
words, counts = zip(*most_common_words)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), palette='viridis')
plt.title('20 Most Frequently Appearing Words')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()


# In[37]:


#Tweets Based on Content Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(tweets_data['cleaned_content'])

kmeans = KMeans(n_clusters=5, random_state=42)
tweets_data['Cluster'] = kmeans.fit_predict(tfidf_matrix)

print(tweets_data[['Content', 'Cluster']].head())


# In[40]:


#Positive VS Negative Sentiment
sentiment_interaction = tweets_data.groupby('sentiment_label')[['Likes', 'Retweets', 'Replies']].mean()

sentiment_interaction.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange', 'green'])
plt.title('Average Interactions Based on Sentiment')
plt.ylabel('Average Interaction')
plt.xticks(rotation=0)
plt.show()

