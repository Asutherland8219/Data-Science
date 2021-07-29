import my_secrets
import tweepy

auth = tweepy.OAuthHandler(my_secrets.apikey, my_secrets.api_s_key)

auth.set_access_token(my_secrets.access_token, my_secrets.access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

import pandas as pd

senators_df = pd.read_csv('Workbook_ex\Datasets\senators.csv')

senators_df['TwitterID'] = senators_df['TwitterID'].astype(str)

print(senators_df.head())

pd.options.display.max_columns = 6

''' Connect to Mongo DB '''
from pymongo import MongoClient

atlas_client = MongoClient(my_secrets.mongo_connection_string)

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
import time

from geopy import OpenMapQuest
from state_codes import state_codes

geo = OpenMapQuest(api_key= my_secrets.consumer_key)
states = tweet_counts_df.State.unique()

states.sort()

locations = []

for state in states:
    processed = False
    delay = .1
    while not processed:
        try:
            locations.append(
                geo.geocode(state_codes[state] + ', USA'))
            print(locations[-1])
            processed = True
        except: #timed out so wait before trying again
            print('OpenMapQuest Service timed out. Waiting...')
            time.sleep(delay)
            delay += .1

# group the tweet counts by state
tweet_counts_by_state = tweet_counts_df.groupby('State', as_index=False).sum()

''' Create the map '''
import folium

usmap = folium.Map(location=[39.8283, -98.5795], zoom_start=4, detect_retina=True, tiles='Stamen Toner')

''' Create map markers for each state '''
sorted_df = tweet_counts_df.sort_values(
    byh='Tweets', ascending=True
)

for index, (name, group) in enumerate(sorted_df.groupby('State')):
    strings = [state_codes[name]]

    for s in group.itertuples():
        strings.append(
            f'{s.Name} ({s.Party}); Tweets: {s.tweets}')
    
    text = '<br>'.join(strings)
    marker = folium.Marker(
        (locations[index].latitude, locations[index].longitude),
        marker.add_to(usmap)
    )

''' Save the file '''
usmap.save('SenatorsTweets.html')


