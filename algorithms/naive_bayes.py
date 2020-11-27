from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
import pandas as pd
import numpy as np

import string, random

stop_words = stopwords.words('english')


def read_file(filename):
    """
    Reads the file using pandas and returns reviews and ratings as array
    :param filename: filename
    :return: review text and rating
    """
    print("reading file...")
    df_hotel = pd.read_csv(filename, encoding="utf-8")
    size = min(10000, len(df_hotel))
    df_hotel = df_hotel.sample(size)
    df_hotel = df_hotel.reset_index(drop=True)
    try:
        # get reviews column for hotel
        reviews_text = df_hotel['reviews.text']
        # get ratings column for hotel
        ratings = df_hotel['reviews.rating']
    except:
        # get reviews column for amazon
        reviews_text = df_hotel['Reviews']
        # get ratings column for amazon
        ratings = df_hotel['Rating']
    return reviews_text, ratings


def pre_process(reviews_text):
    """
    tokenize and cleans the data removing stop words
    :param reviews_text: review sentence
    :return: list of cleaned reviews as tokens
    """
    print("pre-processing text reviews...")
    # tokenize each review and store in list
    hotel_tokens = [word_tokenize(text) for text in reviews_text]

    if "not" in stop_words:
        stop_words.remove("not")

    # preprocess the data
    cleaned_tokens_list = []
    for tokens in hotel_tokens:
        cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    return cleaned_tokens_list


def remove_noise(review_tokens, stop_words=()):
    """
    Removes punctuations, lemmatize the tokens, and performs pos tagging
    :param review_tokens: list of tokens of a review
    :param stop_words: stop words from nltk library
    :return: cleaned list of tokens for each review
    """
    # print("noise")
    NoWord = ["''", "``", '...']
    cleaned_tokens = []
    lemmatizer = WordNetLemmatizer()
    # perform pos tagging on each token of a review
    for token, tag in pos_tag(review_tokens):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        # lemmatize the token
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words and token not in NoWord:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def create_data(cleaned_tokens_list, ratings):
    """
    labels reviews as positive, negative or neutral based on rating and splits into train and test set
    :param cleaned_tokens_list: cleaned list of tokens for each review
    :param ratings: rating of the corresponding review
    :return: train and test data
    """
    print("creating train and test data...")
    # model requires token as (dictionary of token, True)
    tokens_for_model = get_reviews_for_model(cleaned_tokens_list)

    # create dataset labelling each token list as positive or negative based on rating
    dataset = []
    for ind, hotel_dict in enumerate(tokens_for_model):
        if ratings[ind] > 3.8:
            dataset.append((hotel_dict, 'Positive'))
        elif ratings[ind] > 3.0:
            dataset.append((hotel_dict, 'Neutral'))
        # elif ratings[ind] > 2:
        #     dataset.append((hotel_dict, 'Neutral'))
        # elif ratings[ind] > 1:
        #     dataset.append((hotel_dict, 'Negative'))
        else:
            dataset.append((hotel_dict, 'Negative'))
    # shuffle the dataset for random sampling
    random.shuffle(dataset)
    # sample the data into training and testing data
    train_data = dataset[:7000]
    test_data = dataset[7000:]
    return train_data, test_data

def get_reviews_for_model(cleaned_tokens_list):
    """
    create dictionary of token:True for each token
    :param cleaned_tokens_list: cleaned list of tokens for each review
    :return: dictionary of token in each review required to train naive classifier model
    """
    # print("get")
    # create dictionary of token:True for each token
    for review_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in review_tokens)


def check_for_custom_reviews(model, custom_reviews):
    """
    test the model on custom reviews
    :param model: trained naive bayes classifier
    :param custom_reviews: customized reviews to test
    :return: None
    """
    print("checking custom reviews on trained model...")
    for custom_review in custom_reviews:
        custom_tokens = remove_noise(word_tokenize(custom_review))
        print(model.classify(dict([token, True] for token in custom_tokens)))


def main():
    # read data
    file = input("Enter file name:\n")
    reviews_text, ratings = read_file(file)

    # preprocess the text
    cleaned_tokens_list = pre_process(reviews_text)

    # create train and test data
    train_data, test_data = create_data(cleaned_tokens_list, ratings)

    # train the classifier on training data
    model = NaiveBayesClassifier.train(train_data)

    # test the accuracy using test data
    print("Accuracy is:", (classify.accuracy(model, test_data))*100,str("%"))
    # Top 10 Features used to classify
    print(model.show_most_informative_features(10))

    # Test for custom review
    custom_reviews = ["The food was delicious. I would like to order again",
                      "The rooms were not clean and the service was very bad",
                      'The hotel was expensive but overall a nice experience',
                      'The quantity of the food was not enough but tasted good']
    # custom_reviews = ["Very clean set up and easy set up.", "The camera quality was not that great", 'There was
    # only one little blemish on the side,but who cares as long as the phone is fullly functional', "I'm really
    # disappointed about my phone and service"]

    check_for_custom_reviews(model, custom_reviews)


if __name__ == '__main__':
    main()
