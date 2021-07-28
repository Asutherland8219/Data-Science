import tweepy, secrets

auth = tweepy.OAuthHandler(secrets.apikey, secrets.api_s_key)

auth.set_access_token(secrets.access_token, secrets.access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

import pandas as pd 
senators_df = pd.read_csv('Workbook_ex\Datasets\senators.csv')

senators_df['TwitterID'] = senators_df['TwitterID'].astype(str)

print(senators_df.head())

pd.options.display.max_columns = 6

''' Connect to Mongo DB '''
from pymongo import MongoClient
atlas_client = MongoClient(secrets.mongo_connection_string)

db = atlas_client.senators

''' Setting up the tweet stream '''
from tweetlistener import TweetListener

tweet_limit = 10000
twitter_stream = tweepy.Stream(api.auth, TweetListener(api, db, tweet_limit))

twitter_stream.filter(track=senators_df.TwitterHandle.tolist(), follow=senators_df.TwitterID.tolist())

db.tweets.create_index([('$**', 'text')])

tweet_counts = []

for senator in senators_df.TwitterHandle:
    tweet_counts.append(db.tweets.count_documents(
        {"$text": {"$search": senator}}))

''' Show the tweets '''
tweet_counts_df = senators_df.assign(Tweets=tweet_counts)

tweet_counts_df.sort_values(by='Tweets', ascending=False).head(10)

''' Get state and plot location '''
from geopy import OpenMapQuest
import time 
from state_codes import state_codes
