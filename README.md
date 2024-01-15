# YouTube Comment Sentiment Analysis

This project performs sentiment analysis on about 4800 scraped YouTube comments of a Music Video to understand and visualize the underlying sentiment.

## Data Collection
Comments were scraped from a YouTube video using Selenium
Comments were extracted and stored in a Pandas DataFrame

## Data Cleaning & Preprocessing
- Removed empty and incomplete comments
- Cleaned text by removing special characters, URLs, etc.
- Tokenized comments
- Removed stopwords
- Lemmatized words

## Sentiment Analysis
- Used VADER sentiment analysis to assign a compound sentiment score
- Classified comment sentiment as positive, negative or neutral based on score

## Analysis
1. To count and percentage breakdown of sentiment categories
2. To extract and display most positive and negative comments
3. To identify meaningful sentiment words and their frequencies
4. To understand frequently used sentiment adjectives

## Topic Modeling
- Applied Latent Dirichlet Allocation (LDA) to discover topics in the comments

## Named Entity Recognition
- Extracted named entities like people, organizations, locations etc.

## Future Improvements
- Collect and analyze comments from multiple videos of same kind
- Add user and engagement data to provide more context
- Use word vectors to enhance analysis
- Implement classification model for comment sentiments
