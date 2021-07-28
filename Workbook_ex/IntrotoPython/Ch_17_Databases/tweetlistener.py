import json 
import tweepy

class TweetListener(tweepy.StreamListener):
    "Handles incoming tweet stream"

    def __init__(self, api, database, limit= 10000):
        " Create instance variables for the tweets "
        self.db = database
        self.tweet_count = 0
        self.TWEET_LIMIT = limit 
        super().__init__(api)

        def on_connect(self):
            "Called when connection attempt works, applicaton is ready to roll "
            print("Successfully connected to Twitter")

        def on_data(self,data):
            "Called when a new tweet is pushed"
            self.tweet_count += 1 
            json_data = json.loads(data)
            self.db.tweets.insert_one(json_data)
            print(f' Screen name: {json_data["user"]["name"]}')
            print(f' Created at: {json_data["created_at"]}')
            print(f' Tweets Received: {self.tweet_count}')

            # if the tweet limit is reached, terminate streaming
            return self.tweet_count != self.TWEET_LIMIT

        def on_error(self, status):
            print(status)
            return True
      
      