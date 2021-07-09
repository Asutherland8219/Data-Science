from textblob import TextBlob
import tweepy
import secrets

import wordcloud

auth = tweepy.OAuthHandler(secrets.apikey, secrets.api_s_key)

auth.set_access_token(secrets.access_token, secrets.access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# wait on rate limit = wait 15 mins each time it hits api limit
# wait on rate limit notify just enables it to show in shell 
def print_tweets(tweets):
    ''' For each status object in the tweets, display the screen and text. Then use textblob to translate to english'''
    for tweet in tweets:
        print(f'{tweet.user.screen_name}:', end=' ')

        if 'en' in tweet.lang:
            print(f'{tweet.text}')
        elif 'und' not in tweet.lang:
            print(f'\n ORIGINAL: {tweet.text}')
            print(f'TRANSLATED: {TextBlob(tweet.text).translate()}')


tweets = api.search(q='Oil', count=3)
print_tweets(tweets)

''' Search for a hashtag '''
tweets = api.search(q='#itscominghome', count=10)
print_tweets(tweets)


''' Search for trends available; please note this is trendy places not topics '''
trends_available = api.trends_available()
print(len(trends_available))

''' We can narrow down to a specific place or hashtag  '''
world_trends = api.trends_place(id= 1)

trends_list = world_trends[0]['trends']

trends_list = [t for t in trends_list if t['tweet_volume']]

from operator import itemgetter
trends_list.sort(key=itemgetter('tweet_volume'), reverse=True)

for trend in trends_list[:5]:
    print(trend['name'])

''' Word cloud '''
topics = {}

for trend in trends_list:
   topics[trend['name']] = trend['tweet_volume']

print(topics)

from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=900, prefer_horizontal=0.5, min_font_size=10, colormap='prism', background_color='white')

wordcloud = wordcloud.fit_words(topics)

wordcloud = wordcloud.to_file('TrendingTwitterJuly2021.png')

