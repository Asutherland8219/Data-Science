import tweepy, secrets

auth = tweepy.OAuthHandler(secrets.apikey, secrets.api_s_key)

auth.set_access_token(secrets.access_token, secrets.access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

''' Connect to Mongo DB '''
from pymongo import MongoClient
atlas_client = MongoClient(secrets.mongo_connection_string)