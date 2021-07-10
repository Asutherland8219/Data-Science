from geopy import OpenMapQuest
import mapquest_secrets
import time
import tweepy
import secrets

def get_tweet_content(tweet, location=False):
    """Return a tweet dictionary"""
    fields = {}
    fields['screen_name'] = tweet.user.screen_name

    try:
        fields['text'] = tweet.extended_tweet.full_text
    except:
        fields['text'] = tweet.text

    if location:
        fields['location'] = tweet.user.location
    
    return fields

def get_geocodes(tweet_list):
    """"Gets position for each tweet in coordinates"""
    print('Getting Coordinates for tweet locations...')

    geo = OpenMapQuest(api_key= mapquest_secrets.consumer_key)
    bad_locations = 0

    for tweet in tweet_list:
        processed = False
        delay = .1
        while not processed:
            try: # get the coordinates
                geo_location =geo.geocode(tweet['location'])
                processed = True
            
            except: # if times out try again
                print('OpenMapQuest service timed out. Waiting...')
                time.sleep(delay)
                delay += 1
            
            if geo_location:
                tweet['latitude'] = geo_location.latitude
                tweet['longitude'] = geo_location.longitude
            else:
                bad_locations += 1 #invalid location
    print('Done Geocoding')
    return bad_locations

def get_API():
    auth = tweepy.OAuthHandler(secrets.apikey, secrets.api_s_key)

    auth.set_access_token(secrets.access_token, secrets.access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    return api


