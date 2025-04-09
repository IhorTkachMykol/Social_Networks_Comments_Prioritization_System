from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle


# Load the model from the file
with open('src/spam_classifier/trained_models/pipe_spam_cls_model.pkl', 'rb') as f:
    spam_classifier_model = pickle.load(f)


def detect_spam(text):
  # TODO: Ihor fix this docstring
  '''
  1 if spam
  
  This function will classificate keywords (1) and statements or
  question (0) comments

  Args::
      query (str): user's comment.
  Returns:
      0 or 1: 0 if input query is statement or question, 1 if it is a keyword
  '''

  # Predict spam
  return spam_classifier_model.predict([f"{text}"])



print(*detect_spam('i love this so much. AND also I Generate Free Leads on Auto Pilot &amp; You Can  Too! http://www.MyLeaderGate.com/moretrafficï»¿'))