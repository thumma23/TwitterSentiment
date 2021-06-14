from build_test_train_sets import tweet_sentiment
from preprocess_tweets import PreProcessTweets
from language_processor import build_vocabulary
import nltk


api_obj = tweet_sentiment()
pp_tweet = PreProcessTweets()
training_set = api_obj.buildTrainingSet()

clean_train_tweets = pp_tweet.add_clean_tweet_to_list(training_set)
lang_obj = build_vocabulary(clean_train_tweets)

vocab = lang_obj.remove_duplicates()
training_feature_vector = nltk.classify.apply_features(lang_obj.build_feature_vector, clean_train_tweets)

NBayesClassifier = nltk.NaiveBayesClassifier.train(training_feature_vector)