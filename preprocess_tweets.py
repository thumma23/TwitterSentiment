import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords

class PreProcessTweets:
    def __init__(self):
        self.stop_words = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    
    def clean_tweet(self, tweets):
        tweets = tweets.lower()
        tweets = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweets)
        tweets = re.sub('@[^\s]+', 'AT_USER', tweets)
        tweets = re.sub(r'#([^\s]+)', r'\1', tweets)
        tweets = word_tokenize(tweets)
        return [word for word in tweets if word not in self.stop_words]

    def add_clean_tweet_to_list(self, tweets):
        processedTweets= []
        for tweet in tweets:
            processedTweets.append((self.clean_tweet(tweet["text"]), tweet["label"]))
        return processedTweets