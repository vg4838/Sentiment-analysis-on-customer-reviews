import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def get_sentiment(prediction):
    if prediction == 1:
        sentiment = "Very Negative"
    elif prediction == 2:
        sentiment = "Negative"
    elif prediction == 3:
        sentiment = "Neutral"
    elif prediction == 4:
        sentiment = "Positive"
    elif prediction == 5:
        sentiment = "Very Positive"
    return sentiment


def run_k_nearest_neighbors():
    # Read file into dataframe
    df = pd.read_csv("sample_amazon_reviews.csv")

    # generating one row
    df = df.sample(n=10000)

    # Create a label encoder
    label_encoder = preprocessing.LabelEncoder()

    # Convert the string labels into numbers
    # df = df.apply(lambda x: label_encoder.fit_transform(x))

    # Choose the independent and dependent variables
    X = df.Reviews
    y = df.Rating

    # Use CountVectorizer to convert text into tokens/features
    vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=4)

    # Split the dataset in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Use training data to transform text into counts of features for each message
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Create an error list
    error = []

    # Calculate the error rate for K values between 1 and 20
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_dtm, y_train)
        pred_i = knn.predict(X_test_dtm)
        error.append(np.mean(pred_i != y_test))

    # Plot the error rate
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model 
    knn.fit(X_train_dtm, y_train)

    # Make prediction
    y_train_pred = knn.predict(X_train_dtm)
    y_test_pred = knn.predict(X_test_dtm)

    # Make and print a confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix of training set: ", cm_train)
    print("Confusion Matrix of testing set: ", cm_test)

    # Print the accuracy
    print("K-NN model accuracy(in %) of training set:", accuracy_score(y_train, y_train_pred) * 100)
    print("K-NN model accuracy(in %) of test set:", accuracy_score(y_test, y_test_pred) * 100)

    # Print the classification report
    print(classification_report(y_test, y_test_pred))
    '''
    # Perform SVM model
    SVM = LinearSVC()
    SVM.fit(X_train_dtm, y_train)
    y_pred = SVM.predict(X_test_dtm)
    print('\nSupport Vector Machine')
    print("Accuracy Score: ", accuracy_score(y_test, y_pred) * 100, '%', sep='')
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred), sep='\n')

    # Perform Logistic Regression model
    LR = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR.fit(X_train_dtm, y_train)
    y_pred = LR.predict(X_test_dtm)
    print('\nLogistic Regression')
    print('Accuracy Score: ', accuracy_score(y_test, y_pred) * 100, '%', sep='')
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred), sep='\n')
    '''
    # Use training data to transform text into counts of features for each message
    trainingVector = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=5)
    trainingVector.fit(X)
    X_dtm = trainingVector.transform(X)
    knn_complete = KNeighborsClassifier(n_neighbors=5)
    knn_complete.fit(X_dtm, y)
    #LR_complete = LogisticRegression(solver='lbfgs', max_iter=1000)
    #LR_complete.fit(X_dtm, y)
	
    # Create tags
    # tags = ['Positive', 'Neutral', 'Negative']
    tags = ['Very Negative', 'Negative', 'Very Positive', 'Positive', 'Neutral']

    # Get 5 random reviews
    sample_data = X_test.sample(n=5, random_state=2)

    # Test the algorith using the test set
    for review in sample_data:
        review = [review]
        test_dtm = trainingVector.transform(review)
        pred_label = knn_complete.predict(test_dtm)
        print(review, " is predicted ", get_sentiment(pred_label[0]))


def main():
    run_k_nearest_neighbors()


main()
