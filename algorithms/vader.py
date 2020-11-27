import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import numpy as np
import pandas as pd
import random
from scipy.stats import mode

def preprocess_data():
    """
    Purpose: preprocess text (tokenize, stem, and remove stopwords)
    """

    # Open and read file
    # Read file
    print("Reading file... \n")
    data = pd.read_csv("./sample_hotel1_reviews.csv")

    # Get the review text and title columns from the hotel datasets
    text_data = data["reviews.text"].to_numpy()
    title_data = data["reviews.title"].to_numpy()

    # Get the review from the Amazon dataset
    #review_data = data["Reviews"].to_numpy()

    # Shuffle data
    random.Random(3).shuffle(text_data)
    random.Random(3).shuffle(title_data)
    #random.Random(3).shuffle(review_data)

    # Create English stop words list
    stop_words = set(stopwords.words('english')) 

    # Instantiate a stemmer
    ps = PorterStemmer()

    # Instantiate a retokenizer
    detokenizer = TreebankWordDetokenizer()

    # Create a list for tokenized documents in loop
    tokenized_text = []

    for text in title_data[:5]:
         # Create lists
        pos, neg, neu, sentiment = [], [], [], []

        # Split the text into sentences
        sentences = sent_tokenize(text) 

        # Lower all the words and tokenize the sentences
        for sentence in sentences:
            sentence = sentence.lower()
            words = word_tokenize(sentence)

            # Remove stop words
            stopped_words = [w for w in words if not w in stop_words] 

            # Stemmanize the words
            stem_words = [ps.stem(word) for word in stopped_words]

            # Retokenize the words into sentences
            new_sentence = detokenizer.detokenize(stopped_words)

            # Calculate sentiment score
            sentiment_score(new_sentence, pos, neg, neu, sentiment)

        # Print the average sentiment score of each review and the most prevalent sentiment
        print("Average negative score:", sum(neg)/len(neg))
        print("Average neutral score:", sum(neu)/len(neu))
        print("Average positive score:", sum(pos)/len(pos))
        print("Prevalent sentiment:", mode(sentiment))
        print("-------------------------------------------------------NEXT COMMENT-------------------------------------------------------------------")


def sentiment_score(sentence, pos, neg, neu, sentiment):
    # Create a SentimentIntensityAnalyzer object. 
    sia_obj = SentimentIntensityAnalyzer() 
  
    # Create a dictionary that contains positive, negative, neutral, and compound scores. 
    sentiment_dict = sia_obj.polarity_scores(sentence) 
      
    # Print the sentiment scores  
    print("Overall sentiment dictionary is: ", sentiment_dict) 
    print(sentence, "was rated as", sentiment_dict['neg']*100, "% Negative") 
    print(sentence, "was rated as", sentiment_dict['neu']*100, "% Neutral") 
    print(sentence, "was rated as", sentiment_dict['pos']*100, "% Positive") 
  
    # Keep track of the sentiment score of each sentence in a review
    neg.append(sentiment_dict['neg']*100)
    neu.append(sentiment_dict['neu']*100)
    pos.append(sentiment_dict['pos']*100)

    print("Sentence Overall Rated As", end = " ") 
  
    # Decide if the sentiment as positive, negative or neutral using the compound score
    if sentiment_dict['compound'] >= 0.05 : 
        print("Positive") 
        sentiment.append("Positive")
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        print("Negative") 
        sentiment.append("Negative")
  
    else : 
        print("Neutral") 
        sentiment.append("Neutral")

    print("")


def main():

    # Execute preprocess_data
    preprocess_data()

main()

