import datetime
import os
import sys

from nltk import re
from textblob import TextBlob
import snscrape.modules.twitter as sntwitter


def clean_text(tweet):
    '''
    Regular expression that removes links and special characters from tweet.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(https?\S+)", " ", tweet).split())


def get_tweet_sentiment(tweet):
    '''
    Calculate the sentiment using TextBlob module
    TextBlog
    '''
    text = clean_text(tweet)
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def get_tweets_sentiment(query, count=10, start=None, end=None, lang='english'):
    '''
    Given a query returns the tweets
    '''
    tweets = []
    try:
        # calls the API to obtain tweets
        fetched_tweets = get_tweets(
            query,
            count=count,
            start=start,
            end=end,
            lang=lang
        )
        # parsing the tweets
        for tweet in fetched_tweets:
            parsed_tweet = {}
            # get the tweet text
            parsed_tweet['text'] = tweet
            # get the sentiment for the tweet's text
            parsed_tweet['sentiment'] = get_tweet_sentiment(tweet)
            tweets.append(parsed_tweet)
        return tweets
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(str(e), fname, exc_tb.tb_lineno)
        return tweets


def get_tweets(query, count=10, start=None, end=None, lang='english'):
    '''
    Restituisce i tweet data una query particolare
    '''
    if not start:
        end = datetime.date.today()
        start = (end - datetime.timedelta(days=30))
    tweets = []
    query = f'{query} since:{start} until:{end}'
    try:
        # calls the API to obtain tweets
        fetched_tweets = sntwitter.TwitterSearchScraper(query).get_items()
        # parsing the tweets
        for i, tweet in enumerate(fetched_tweets):
            if i > count:
                return tweets
            if tweet.content:
                # add the tweet to our list and avoid retweets
                if tweet.retweetCount > 0:
                    if tweet.content not in tweets:
                        tweets.append(tweet.content)
                else:
                    tweets.append(tweet.content)
        return tweets
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(str(e), fname, exc_tb.tb_lineno)
        return tweets


if __name__ == '__main__':
    query = "donald trump"
    count = 100
    for tweet in get_tweets_sentiment(query, count):
        print(tweet['sentiment'], tweet['text'])
