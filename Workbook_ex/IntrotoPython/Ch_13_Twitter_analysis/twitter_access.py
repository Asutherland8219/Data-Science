import tweepy
import secrets

auth = tweepy.OAuthHandler(secrets.apikey, secrets.api_s_key)

auth.set_access_token()