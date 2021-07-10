''' Get the location of tweets and map them '''
# we need the mapquest api for this
from folium.map import Popup
import mapquest_secrets
from tweetutilities import get_API, get_geocodes
from locationlistener import LocationListener
import tweepy


query = 'Pizza'
amt = int(input('How many?'))
api = get_API()

tweets = []

counts = {'total_tweets': 0, 'locations': 0}

location_listener = LocationListener(api, counts_dict=counts, tweets_list=tweets, topic=query, limit=amt)

stream = tweepy.Stream(auth= api.auth, listener=location_listener)

stream.filter(track= query, languages=['en'], is_async=False)

print(counts)

bad_locations = get_geocodes(tweets)
print(bad_locations)

# cleaning the data 
import pandas as pd 
df = pd.DataFrame(tweets)
df = df.dropna()

# create the map and popup markets
import folium
worldmap = folium.Map()

for t in df.itertuples():
    text = ': '.join([t.screen_name, t.text])
    popup = folium.Popup(text, parse_html=True)
    marker = folium.Marker((t.latitude, t.longitude), popup=popup)
    marker.add_to(worldmap)

worldmap.save('tweet_map.html')
