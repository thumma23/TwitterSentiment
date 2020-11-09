import nltk

class build_vocabulary:
    def __init__(self, clean_train_tweets):
        self.clean_train_tweets = clean_train_tweets

    def remove_duplicates(self):
        vocab = []

        for (words, sentiment) in self.clean_train_tweets:
            vocab.extend(words)

        word_list = nltk.FreqDist(vocab)
        vocab_list = word_list.keys()
        
        return vocab_list

    def build_feature_vector(self, tweet):
        tweet = set(tweet)
        features = {}
        vocab_list = self.remove_duplicates()
        for word in vocab_list:
            features[word] = (word in tweet)
        return features