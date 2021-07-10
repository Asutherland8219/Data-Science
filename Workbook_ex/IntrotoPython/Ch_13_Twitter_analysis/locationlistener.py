import tweepy
from tweetutilities import get_tweet_content

class LocationListener(tweepy.StreamListener):

    def __init__(self, api, counts_dict, tweets_list, topic, limit=10):
        """Configure the listener"""
        self.tweets_list = tweets_list
        self.counts_dict = counts_dict
        self.topic = topic
        self. TWEET_LIMIT = limit
        super().__init__(api)


    def on_status(self, status):
        tweet_data = get_tweet_content(status, location=True)


    #ignore RT
        if (tweet_data['text'].startswith('RT') or self.topic.lower() not in tweet_data['text'].lower()):
            return
        
        self.counts_dict['total_tweets'] += 1

    #ignore locationless tweets
        if not status.user.location:
            return

        self.counts_dict['locations'] += 1 # tweet with location
        self.tweets_list.append(tweet_data) # store the tweet   
        print(f'{status.user.screen_name}: {tweet_data["text"]}\n')

    # if tweet limit is reached terminate stream
        return self.counts_dict['locations'] != self.TWEET_LIMIT 

