'''Searches tweets, tallies postitive, neutral and negative sentiment'''
import secrets
import preprocessor as p
import sys
from textblob import TextBlob, sentiments
import tweepy

class SentimentListener(tweepy.StreamListener):

    def __init__(self, api, sentiment_dict, topic, limit=10):
        "Configure the sentiment listener"
        self.sentiment_dict = sentiment_dict
        self.tweet_count = 0
        self.topic = topic
        self.TWEET_LIMIT = limit

        # Set up to remove url and useless words
        p.set_options(p.OPT.URL, p.OPT.RESERVED)
        super().__init__(api)

    def on_status(self, status):
        try:
            tweet_text = status.extended_tweet.full_text
        except:
            tweet_text = status.text

        #ignore retweets
        if tweet_text.startswith('RT'):
            return

        tweet_text = p.clean(tweet_text)

        if self.topic.lower() not in tweet_text.lower():
            return
        
        blob = TextBlob(tweet_text)
        if blob.sentiment.polarity > 0:
            sentiment = '+'
            self.sentiment_dict['positive'] += 1
        elif blob.sentiment.polarity == 0:
            sentiment = ' '
            self.sentiment_dict['neutral'] += 1
        else:
            sentiment = '-'
            self.sentiment_dict['negative'] += 1

        print(f'{sentiment} {status.user.screen_name}: {tweet_text}\n')

        self.tweet_count += 1 # tracking tweets processed

        #if limit is reached return false to stop streaming
        return self.tweet_count != self.TWEET_LIMIT

def main():
    auth = tweepy.OAuthHandler(secrets.apikey, secrets.api_s_key)

    auth.set_access_token(secrets.access_token, secrets.access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    #Create the listener
    search_key = input("What are you looking for? ")
    limit = int(input("How many tweets do you want?"))# how many tweets do we want?
    sentiment_dict = {'positive': 0, 'neutral': 0, 'negative': 0}
    sentiment_listener = SentimentListener(api, sentiment_dict, search_key, limit)

    stream = tweepy.Stream(auth=api.auth, listener=sentiment_listener)

    stream.filter(track=[search_key], languages=['en'], is_async=False)

    print(f'Tweet sentiment for "{search_key}"')
    print('Positive:', sentiment_dict['positive'])
    print(' Neutral:', sentiment_dict['neutral'])
    print('Negative:', sentiment_dict['negative'])


if __name__ == '__main__':
    main()
