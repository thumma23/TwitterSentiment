from authentication_info import API_key, API_key_secret, Access_token, Access_token_secret
import tweepy

class tweet_sentiment():
    def __init__(self):
        self.auth = tweepy.OAuthHandler(API_key, API_key_secret) 
        self.auth.set_access_token(Access_token, Access_token_secret)
        self.api = tweepy.API(self.auth)

    def buildTestSet(self, query):
        try:
            tweets_fetched = self.api.search(q = query, count = 100)
            print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + query)
            return [{"text": status.text, "label": None} for status in tweets_fetched]
        except:
            print("Unfortunately, something went wrong..")
            return None

    def buildTrainingSet(self):
        corvus_data = []
        training_set = []
        corvus_file = open("corvus.txt", "r")

        for line in corvus_file:
            values = line.split(',')
            counter = 0
            for word in values:
                if(word[1] == '1'):
                    word = word[1:-2]
                else:
                    word = word[1:-1]
                values[counter] = word
                counter = counter + 1
            corvus_data.append({"topic": values[0], "label": values[1], "tweet_id": values[2]})
        
        corvus_file.close()

        for tweet in corvus_data:
            try:
                status = self.api.get_status(tweet["tweet_id"])
                tweet["text"] = status.text
                training_set.append(tweet)
            except:
                continue
            
        return training_set