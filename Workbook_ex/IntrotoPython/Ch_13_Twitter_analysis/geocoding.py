''' Get the location of tweets and map them '''
# we need the mapquest api for this
import mapquest_secrets
from tweetutilities import get_API
from locationlistener import LocationListener

api = get_API()

tweets = []

counts = {'total_tweets': 0, 'locations': 0}

