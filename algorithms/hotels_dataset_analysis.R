library(dplyr)

# Read the file
hotels <- read.csv(file = "./hotel_reviews.csv")

# Select pertinent attributes
hotel_features <- select(hotels, id, name, reviews.rating, reviews.text, reviews.title)

# Remove rows that have at least one empty row
cleansed_hotel_features <- hotel_features[rowSums(is.na(hotel_features)) == 0,]

# Shuffle data
shuffled_hotel_features <- cleansed_hotel_features[sample(nrow(cleansed_hotel_features)),]

# Get a random sample of 10000 rows
sample_hotels <- shuffled_hotel_features[sample(nrow(shuffled_hotel_features), 10000), ]

# Write sample to a file
write.csv(sample_hotels, file = "sample_hotels_reviews.csv")

# Before taking a sample, We removed all the rows with at least one empty cell and shuffle the data.
# The sample consists of 10000 columns and four attributes.
# The attributes are id, name, reviews.rating, reviews.text, reviews.title. 
# id identifies a review. name refers to the hotel. reviews.rating is the number of stars attributed to the hotel.
# reviews.text is the customer's opinions. reviews.title is the summary of the review.text.
