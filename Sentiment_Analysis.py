#!/usr/bin/env python3
# -*- coding: utf-8 -*-

pip install selenium

import time
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

# Set Chrome options
chrome_options = Options()
# Add any additional options as needed

# Specify the path to the ChromeDriver executable
chrome_driver_path = "/Users/jonathangabriel/Downloads/chromedriver_mac64"

# Create a new Service instance
service = Service(chrome_driver_path)

# Create a new ChromeDriver instance
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the YouTube video
video_url = "https://www.youtube.com/watch?v=DrNtuAgwWgQ"

# Maximize the browser window (optional)
driver.maximize_window()

# Navigate to the YouTube video URL
driver.get(video_url)

# Scroll down to load more comments
for item in range(50):
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(2)  # Adjust the wait time as needed

# Find and extract the comments
comments = driver.find_elements(By.CSS_SELECTOR, "#content")

# Store the comments in the data list
data = [comment.text for comment in comments]

# Output the comments
for comment in data:
    print(comment)

# Remember to quit the driver when you're done
driver.quit()


import pandas as pd   
df = pd.DataFrame(data, columns=['comment'])
df.head(400)

# Remove first and second comments
df = df.iloc[2:]

# Remove empty comments
df = df[df['comment'].notna()]

# Remove last 6 comments
df = df.iloc[:-6]

# Reset the index
df = df.reset_index(drop=True)

df

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models

nltk.download('stopwords')


# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()



def preprocess_comment(comment):
    comment = re.sub(r"http\S+|www\S+|https\S+", "", comment)  # Remove URLs
    comment = re.sub(r"[^\w\s]", "", comment)  # Remove special characters
    tokens = word_tokenize(comment)  # Tokenize the comment
    stop_words = set(stopwords.words("english"))  # Get English stopwords
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    return tokens


df['processed_comment'] = df['comment'].apply(preprocess_comment)


import nltk
nltk.download('vader_lexicon')


import nltk
nltk.download('punkt')


from nltk.sentiment import SentimentIntensityAnalyzer

# Create an instance of the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Iterate over the comments and perform sentiment analysis
for comment in df['processed_comment']:
    comment_text = ' '.join(comment)  # Convert the preprocessed comment back to a string
    sentiment_scores = sia.polarity_scores(comment_text)
    sentiment = sentiment_scores['compound']  # Extract the compound sentiment score

    # Determine sentiment label based on the compound score
    if sentiment >= 0.05:
        sentiment_label =  'Positive'
    elif sentiment <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    # Print the comment, sentiment score, and sentiment label
    print(f"Comment: {comment_text}")
    print(f"Sentiment Score: {sentiment_scores['compound']}")
    print(f"Sentiment Label: {sentiment_label}")
    print("----------")


from nltk.sentiment import SentimentIntensityAnalyzer

# Create an instance of the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize lists to store positive and negative comments
positive_comments = []
negative_comments = []

# Iterate over the comments and perform sentiment analysis
for comment in df['processed_comment']:
    comment_text = ' '.join(comment)  # Convert the preprocessed comment back to a string
    sentiment_scores = sia.polarity_scores(comment_text)
    sentiment = sentiment_scores['compound']  # Extract the compound sentiment score

    # Determine sentiment label based on the compound score
    if sentiment >= 0.05:
        positive_comments.append((sentiment, comment_text))
    elif sentiment <= -0.05:
        negative_comments.append((sentiment, comment_text))

# Sort the positive and negative comments based on sentiment score
positive_comments.sort(reverse=True)
negative_comments.sort()

# Get the top 10 positive and negative comments
top_positive_comments = positive_comments[:20]
top_negative_comments = negative_comments[:20]

# Print the top 10 positive comments
print("Top 20 Positive Comments:")
for sentiment, comment in top_positive_comments:
    print(f"Sentiment Score: {sentiment}")
    print(f"Comment: {comment}")
    print("----------")

# Print the top 10 negative comments
print("Top 20 Negative Comments:")
for sentiment, comment in top_negative_comments:
    print(f"Sentiment Score: {sentiment}")
    print(f"Comment: {comment}")
    print("----------")


import matplotlib.pyplot as plt

# Initialize counters for positive, negative, and neutral comments
positive_count = 0
negative_count = 0
neutral_count = 0

# Iterate over the comments and perform sentiment analysis
for comment in df['processed_comment']:
    comment_text = ' '.join(comment)  # Convert the preprocessed comment back to a string
    sentiment_scores = sia.polarity_scores(comment_text)
    sentiment = sentiment_scores['compound']  # Extract the compound sentiment score

    # Determine sentiment label based on the compound score and update the respective count
    if sentiment >= 0.05:
        positive_count += 1
    elif sentiment <= -0.05:
        negative_count += 1
    else:
        neutral_count += 1

# Create a bar chart to visualize the comment counts
sentiment_labels = ['Positive', 'Negative', 'Neutral']
comment_counts = [positive_count, negative_count, neutral_count]

plt.bar(sentiment_labels, comment_counts)
plt.xlabel('Sentiment')
plt.ylabel('Comment Count')
plt.title('Count of Positive, Negative, and Neutral Comments')
plt.show()


import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import words as english_words

# Initialize counters for positive, negative, and neutral comments
positive_count = 0
negative_count = 0
neutral_count = 0

# Get the set of English stopwords
english_stopwords = set(stopwords.words("english"))

# Get the set of English words
english_word_set = set(english_words.words())

# Iterate over the comments and perform sentiment analysis
for comment in df['processed_comment']:
    comment_text = ' '.join(comment)  # Convert the preprocessed comment back to a string
    sentiment_scores = sia.polarity_scores(comment_text)

    # Remove stopwords and non-English words from the comment text
    comment_words = comment_text.split()
    comment_words = [word for word in comment_words if word.lower() not in english_stopwords and word.lower() in english_word_set]
    comment_text = ' '.join(comment_words)

    sentiment = sentiment_scores['compound']  # Extract the compound sentiment score

    # Determine sentiment label based on the compound score and update the respective count
    if sentiment >= 0.05:
        positive_count += 1
    elif sentiment <= -0.05:
        negative_count += 1
    else:
        neutral_count += 1

# Calculate the total number of comments
total_count = positive_count + negative_count + neutral_count

# Calculate the percentage of each sentiment
positive_percentage = (positive_count / total_count) * 100
negative_percentage = (negative_count / total_count) * 100
neutral_percentage = (neutral_count / total_count) * 100

# Create a bar chart to visualize the comment percentages
sentiment_labels = ['Positive', 'Negative', 'Neutral']
comment_percentages = [positive_percentage, negative_percentage, neutral_percentage]

plt.bar(sentiment_labels, comment_percentages)
plt.xlabel('Sentiment')
plt.ylabel('Comment Percentage')
plt.title('Percentage of Positive, Negative, and Neutral Comments')
plt.show()


from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize stopwords
stop_words = set(stopwords.words("english"))

# Initialize counters for positive, negative, and neutral comments
positive_words = []
negative_words = []
neutral_words = []

# Iterate over the comments and perform sentiment analysis
for comment in df['processed_comment']:
    comment_text = ' '.join(comment)  # Convert the preprocessed comment back to a string
    sentiment_scores = sia.polarity_scores(comment_text)
    sentiment = sentiment_scores['compound']  # Extract the compound sentiment score

    # Determine sentiment label based on the compound score and update the respective word list
    if sentiment >= 0.05:
        positive_words.extend([word for word in comment if word not in stop_words and len(word) > 3])
    elif sentiment <= -0.05:
        negative_words.extend([word for word in comment if word not in stop_words and len(word) > 3])
    else:
        neutral_words.extend([word for word in comment if word not in stop_words and len(word) > 3])

# Count the occurrences of words in each sentiment label
positive_word_counts = Counter(positive_words)
negative_word_counts = Counter(negative_words)
neutral_word_counts = Counter(neutral_words)

# Get the top 20 most common words for each sentiment label
top_positive_words = positive_word_counts.most_common(20)
top_negative_words = negative_word_counts.most_common(20)
top_neutral_words = neutral_word_counts.most_common(20)

# Print the top 20 most common words for each sentiment label
print("Top 20 meaningful words in positive sentiment:")
for word, count in top_positive_words:
    print(f"{word}: {count}")

print("\nTop 20 meaningful words in negative sentiment:")
for word, count in top_negative_words:
    print(f"{word}: {count}")

print("\nTop 20 meaningful words in neutral sentiment:")
for word, count in top_neutral_words:
    print(f"{word}: {count}")


import matplotlib.pyplot as plt

# Get the top 20 most common words for each sentiment label
top_positive_words = positive_word_counts.most_common(20)
top_negative_words = negative_word_counts.most_common(20)
top_neutral_words = neutral_word_counts.most_common(20)

# Extract the words and their corresponding counts for plotting
positive_words, positive_counts = zip(*top_positive_words)
negative_words, negative_counts = zip(*top_negative_words)
neutral_words, neutral_counts = zip(*top_neutral_words)

# Plot the top 20 most common words for each sentiment label
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.barh(range(len(positive_words)), positive_counts, align='center', color='green')
plt.yticks(range(len(positive_words)), positive_words)
plt.xlabel('Word Count')
plt.ylabel('Words')
plt.title('Top 20 Meaningful Words in Positive Sentiment')

plt.subplot(1, 3, 2)
plt.barh(range(len(negative_words)), negative_counts, align='center', color='red')
plt.yticks(range(len(negative_words)), negative_words)
plt.xlabel('Word Count')
plt.ylabel('Words')
plt.title('Top 20 Meaningful Words in Negative Sentiment')

plt.subplot(1, 3, 3)
plt.barh(range(len(neutral_words)), neutral_counts, align='center', color='blue')
plt.yticks(range(len(neutral_words)), neutral_words)
plt.xlabel('Word Count')
plt.ylabel('Words')
plt.title('Top 20 Meaningful Words in Neutral Sentiment')

plt.tight_layout()
plt.show()


english_stopwords


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Convert the preprocessed comments into a document-term matrix
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(df['processed_comment'].apply(lambda x: ' '.join(x)))

# Apply LDA to the document-term matrix
lda = LatentDirichletAllocation(n_components=20)  # Specify the number of topics
lda.fit(dtm)

# Get the most important keywords for each topic
feature_names = vectorizer.get_feature_names()
num_top_keywords = 10  # Number of keywords to extract for each topic
topic_keywords = []
for topic_weights in lda.components_:
    top_keyword_indices = topic_weights.argsort()[:-num_top_keywords - 1:-1]
    top_keywords = [feature_names[index] for index in top_keyword_indices]
    topic_keywords.append(top_keywords)

# Print the keywords for each topic
for topic_id, keywords in enumerate(topic_keywords):
    print(f"Topic #{topic_id}: {', '.join(keywords)}")


pos_tagged_comments = [pos_tag(comment) for comment in df['processed_comment']]


import nltk
nltk.download('averaged_perceptron_tagger')


pos_tagged_comments


english_words = set(nltk.corpus.words.words())
adjectives = [word for comment in pos_tagged_comments for word, pos in comment if pos == 'JJ' and word.lower() in english_words]

# Count the frequency of adjectives
adjective_freq = nltk.FreqDist(adjectives)

# Print the most common adjectives and their frequencies
for word, frequency in adjective_freq.most_common():
    print(f"{word}: {frequency}")


from nltk.sentiment import SentimentIntensityAnalyzer

# Create an instance of the sentiment analyzer
sia = SentimentIntensityAnalyzer()

positive_words = []
negative_words = []

# Iterate over the adjective frequency distribution
for word, frequency in adjective_freq.items():
    # Calculate the sentiment score for each word
    sentiment_score = sia.polarity_scores(word)['compound']
    
    # Classify words as positive or negative based on sentiment score
    if sentiment_score >= 0.05:
        positive_words.append(word)
    elif sentiment_score <= -0.05:
        negative_words.append(word)

# Print the list of positive and negative words
print("Positive Words:")
print(positive_words)
print("\nNegative Words:")
print(negative_words)


from nltk.sentiment import SentimentIntensityAnalyzer

# Create an instance of the sentiment analyzer
sia = SentimentIntensityAnalyzer()

positive_words = []
negative_words = []

# Iterate over the adjective frequency distribution
for word, frequency in adjective_freq.items():
    # Calculate the sentiment score for each word
    sentiment_score = sia.polarity_scores(word)['compound']
    
    # Classify words as positive or negative based on sentiment score
    if sentiment_score >= 0.05:
        positive_words.append((word, sentiment_score))
    elif sentiment_score <= -0.05:
        negative_words.append((word, sentiment_score))

# Sort the positive and negative words based on sentiment score
positive_words = sorted(positive_words, key=lambda x: x[1], reverse=True)
negative_words = sorted(negative_words, key=lambda x: x[1], reverse=False)

# Print the list of positive and negative words with sentiment scores
print("Positive Words:")
for word, score in positive_words:
    print(f"{word}: {score}")

print("\nNegative Words:")
for word, score in negative_words:
    print(f"{word}: {score}")


from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist

# Create an instance of the sentiment analyzer
sia = SentimentIntensityAnalyzer()

positive_words = []
negative_words = []

# Iterate over the adjective frequency distribution
for word, frequency in adjective_freq.items():
    # Calculate the sentiment score for each word
    sentiment_score = sia.polarity_scores(word)['compound']
    
    # Classify words as positive or negative based on sentiment score
    if sentiment_score >= 0.05:
        positive_words.extend([word] * frequency)
    elif sentiment_score <= -0.05:
        negative_words.extend([word] * frequency)

# Create frequency distributions for positive and negative words
positive_freq_dist = FreqDist(positive_words)
negative_freq_dist = FreqDist(negative_words)

# Get the top 10 most frequent positive words
top_positive_words = positive_freq_dist.most_common(10)

# Get the top 10 most frequent negative words
top_negative_words = negative_freq_dist.most_common(10)

# Print the top 10 positive words
print("Top 10 Positive Words:")
for word, frequency in top_positive_words:
    print(f"{word}: {frequency}")

print("\n")

# Print the top 10 negative words
print("Top 10 Negative Words:")
for word, frequency in top_negative_words:
    print(f"{word}: {frequency}")


import matplotlib.pyplot as plt

# Get the top 10 positive words and their frequencies
top_positive_words = positive_freq_dist.most_common(10)
positive_words, positive_frequencies = zip(*top_positive_words)

# Get the top 10 negative words and their frequencies
top_negative_words = negative_freq_dist.most_common(10)
negative_words, negative_frequencies = zip(*top_negative_words)

# Create a figure and axes
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))

# Plot the top 10 positive words
ax1.barh(positive_words, positive_frequencies, color='green')
ax1.set_title('Top 10 Positive Words')
ax1.set_xlabel('Frequency')

# Plot the top 10 negative words
ax2.barh(negative_words, negative_frequencies, color='red')
ax2.set_title('Top 10 Negative Words')
ax2.set_xlabel('Frequency')

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


nltk.download('words')


ner_comments = [ne_chunk(comment) for comment in pos_tagged_comments]


import nltk
nltk.download('maxent_ne_chunker')


import nltk
nltk.download('words')

ner_comments


dictionary = corpora.Dictionary(df['processed_comment'])
corpus = [dictionary.doc2bow(comment) for comment in df['processed_comment']]
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary)


lda_model

corpus 


keywords = set()
for comment in df['processed_comment']:
    keywords.update(comment)

topic_keywords = []
for topic in lda_model.print_topics():
    topic_keywords.extend(topic[1].split("+"))

# Print the extracted keywords
print("Keywords:")
print(keywords)
print("\nTopic Keywords:")
print(topic_keywords)


pip install wordcloud


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Convert the preprocessed comments into a document-term matrix
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(df['processed_comment'].apply(lambda x: ' '.join(x)))

# Apply LDA to the document-term matrix
lda = LatentDirichletAllocation(n_components=10)  # Specify the number of topics
lda.fit(dtm)

# Get the most important keywords for each topic
feature_names = vectorizer.get_feature_names()
num_top_keywords = 5  # Number of keywords to extract for each topic
topic_keywords = []
for topic_weights in lda.components_:
    top_keyword_indices = topic_weights.argsort()[:-num_top_keywords - 1:-1]
    top_keywords = [feature_names[index] for index in top_keyword_indices]
    topic_keywords.append(top_keywords)


topic_keywords


import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

df = pd.DataFrame(data, columns=['comment'])

# Remove mentions
df['tidy_comment'] = np.vectorize(remove_pattern)(df['comment'], "@[\w]*")

# Remove special characters and numbers
df['tidy_comment'] = df['tidy_comment'].str.replace("[^a-zA-Z#]", " ")

# Tokenization
tokenized_comment = df['tidy_comment'].apply(lambda x: x.split())

# Remove stopwords
tokenized_comment = tokenized_comment.apply(lambda x: [word for word in x if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
tokenized_comment = tokenized_comment.apply(lambda x: [stemmer.stem(i) for i in x])

# Lemmatization
lemmatizer = WordNetLemmatizer()
tokenized_comment = tokenized_comment.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

# Join the tokens back into sentences
for i in range(len(tokenized_comment)):
    tokenized_comment[i] = ' '.join(tokenized_comment[i])

df['tidy_comment'] = tokenized_comment

# Create word cloud
all_words = ' '.join([text for text in df['tidy_comment']])

# Extract meaningful words
meaningful_words = []
for word in all_words.split():
    if len(word) > 3:  # Filter out short words
        meaningful_words.append(word)

from collections import Counter

# Count word frequency
word_freq = Counter(meaningful_words)

# Get the most common words and their frequencies
top_words = word_freq.most_common(30)  # Change the number as desired

# Extract words and frequencies
words = [word for word, freq in top_words]
frequencies = [freq for word, freq in top_words]

# Create bar chart
plt.figure(figsize=(10, 7))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 30 Most Common Words')
plt.xticks(rotation=45)
plt.show()


meaningful_words


import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with word frequencies
word_freq = df['tidy_comment'].str.split(expand=True).stack().value_counts().reset_index()
word_freq.columns = ['word', 'frequency']

# Set the number of words to display
num_words = 20

# Select the top 'num_words' words by frequency
top_words = word_freq.head(num_words)

# Create a horizontal bar chart
plt.figure(figsize=(10, 7))
plt.barh(top_words['word'], top_words['frequency'])
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top {} Words'.format(num_words))
plt.gca().invert_yaxis()  # Invert y-axis to display highest frequency words on top
plt.show()


import nltk
nltk.download('wordnet')


import nltk
nltk.download('omw-1.4')


import matplotlib.font_manager as fm

# Retrieve the list of TrueType font files
font_files = fm.findSystemFonts()

# Print the font file paths
for font_file in font_files:
    print(font_file)


get_ipython().system('python3 -m PIL')


get_ipython().system('pip install pillow')
