from build_test_train_sets import tweet_sentiment
from preprocess_tweets import PreProcessTweets
from language_processor import build_vocabulary
import nltk

query = input("Query to Search: ")

api_obj = tweet_sentiment()
pp_tweet = PreProcessTweets()
test_set = api_obj.buildTestSet(query)
training_set = api_obj.buildTrainingSet()

clean_train_tweets = pp_tweet.add_clean_tweet_to_list(training_set)
clean_test_tweet = pp_tweet.add_clean_tweet_to_list(test_set)
lang_obj = build_vocabulary(clean_train_tweets)

vocab = lang_obj.remove_duplicates()
training_feature_vector = nltk.classify.apply_features(lang_obj.build_feature_vector, clean_train_tweets)

NBayesClassifier = nltk.NaiveBayesClassifier.train(training_feature_vector)
NBResultLabels = [NBayesClassifier.classify(lang_obj.build_feature_vector(tweet[0])) for tweet in clean_test_tweet]

positive_percentage = 100 * (NBResultLabels.count('positive') / len(NBResultLabels))
negative_percentage = 100 * (NBResultLabels.count('negative') / len(NBResultLabels))
neutral_percentage = 100 * (NBResultLabels.count('neutral') / len(NBResultLabels))

print("Positive Tweet Percentage: " + str(positive_percentage))
print("Negative Tweet Percentage: " + str(negative_percentage))
print("Neutral Tweet Percentage: " + str(neutral_percentage))