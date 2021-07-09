import tweepy
import secrets

auth = tweepy.OAuthHandler(secrets.apikey, secrets.api_s_key)

auth.set_access_token(secrets.access_token, secrets.access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# wait on rate limit = wait 15 mins each time it hits api limit
# wait on rate limit notify just enables it to show in shell 

elon = api.get_user('elonmusk')

print(elon.id)

# elon.followers_count
# elon.friends_count

''' If you want to get information about yourself or your account use the following'''
# me = api.me()
# api.home_timeline()

''' The textbook references how to get who someone follows or who follows them. That information is essentially useless so was omitted'''


''' Getting recent Tweets'''
elon_tweets = api.user_timeline(screen_name = 'elonmusk', count= 3)

for tweet in elon_tweets:
    print(f'{tweet.user.screen_name}: {tweet.text}')

