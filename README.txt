README

hotels_dataset_analysis.R
- Use RStudio to cleanse hotel 1 dataset.
- Select pertinent features
- Remove rows that have at least one empty row
- Extract a random sample of 10000 rows into a csv file

vader.py
This file contains three methods: preprocess_data(), sentiment_score(), and main(), as well as several libraries
Libraries
- nltk, vaderSentiment.vaderSentiment, numpy, pandas, scipy.stats

preprocess_data()
- Reads data from a file whose name has to be manually typed
- Extracts text and title reviews
- Performs the followings operations: tokenization, putting words in lowercase, removing stop words, and stemming
- Calls sentiment_score() method
- Prints most frequent sentiment within a corpus

sentiment_score()
- Calculates the polarity and intensity of all the sentences in each review.
- Prints the overall polarity of the reviews

main()
- Calls preprocess_data() method

Execution process
- Make sure the libraries are downloaded using pip or pip3 functions
- Manually type the name of the data file
- Depending on the name of the features in the data file, you will have to change the attributes you extract
  For Hotels datasets 1 and 2, we use "review.text" and "review.title". For the Amazon dataset, we used "Reviews"
- Run the code for each selected feature





------------------------------------------------------------------------------------------------------------------

naive bayes.py
This file contains seven methods: read_file(), pre_process(), remove_noise(), create_data(), get_reviews_for_model(), check_for_custom_reviews() and main(), as well as several libraries
Libraries
- nltk, numpy, panda

read_file(): Reads the file using pandas and returns reviews and ratings as array
preprocess_data(): Tokenizes and cleans the data and removes stop words
remove_noise(): Removes punctuations, lemmatizes the tokens, and performs pos tagging
create_data(): Labels reviews as positive, negative or neutral based on rating and splits into train and test set
get_reviews_for_model(): Creates a dictionary of token:True for each token required for naive bayes classifier
check_for_custom_reviews(): Tests the model on custom reviews
main():start point of program.

Execution process:
Pass the file path of the dataset you want to test asked as input.
The code handles the column name for different datasets.
Also, if you want to provide your custom review for hotels, update the custom reviews array in main() function with your own text.


------------------------------------------------------------------------------------------------------------------
Anmol Jaising

preprocess.py

Reads data from a file whose name has to be manually typed
Extracts text and title reviews
Tokenization, Removing of stop words, stemming, retokinzation
Return a corpus as a list of detockenized words


k nearest neighbors.py

This file contains three methods: get_sentiment(), knn(), and main(), as well as several libraries
Libraries
- nltk, vaderSentiment.vaderSentiment, numpy, panda

get_sentiment(prediction)
- classifies neighbours based on their score


main()
-run_k_nearest_neighbors()

run_k_nearest_neighbors()
- Reads data from a file file whose name has to be manually type
- Calculates the k-nearest neighbour of a predictable variable. Calculates the accuracy of the dataset and polarity of the reviews


Execution process
- Make sure the libraries are downloaded using pip or pip3 functions
- Manually type the name of the data file
- Depending on the name of the features in the data file, you will have to change the attributes you extract.
  For Hotels datasets 1 and 2, we use "review.text" and "review.title". For the Amazon dataset, we used "Reviews"
- Run the code for each selected feature. For Amazon, the preprocess.py has to be run first and then 
  k nearest neighbors.py

