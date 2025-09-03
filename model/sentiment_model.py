import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
reviews_df = pd.read_csv("./data/IMDB Dataset.csv")

# change the sentiment column to numbers (possitive = 0 | negative = 1)
reviews_df["result"] = reviews_df["sentiment"].apply(lambda x: 1 if x == "negative" else 0)

# split data into testing and training data
x_train,x_test,y_train,y_test = train_test_split(reviews_df.review,reviews_df.result,test_size=0.35)

pipeline = Pipeline([
    ("vectorizer",CountVectorizer()),
    ("classifier",MultinomialNB())
])

pipeline.fit(x_train,y_train)
accuracy = pipeline.score(x_test,y_test)
print(f"Accuracy = {accuracy}")

joblib.dump(pipeline,"./model/model.pkl")