import csv

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from twitterscraper_examples import get_tweets_sentiment


def topic_modeling(tweets=None):
    if not tweets:
        tweets = []
    tf_vectorizer = CountVectorizer(
        max_df=2,
        min_df=1,
        max_features=1000,
        stop_words='english'
    )
    tf = tf_vectorizer.fit_transform(tweets)
    tf_feature_names = tf_vectorizer.get_feature_names()
    no_topics = 5
    lda = LatentDirichletAllocation(n_components=no_topics,
                                    max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0).fit(tf)
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([tf_feature_names[i]
                        for i in topic.argsort()[:-10 - 1:-1]]))


def dump_tweets(query, count):
    tweets = get_tweets_sentiment(query=query,
                                  count=count)
    file_path = "data/tweets_%s.csv" % query.replace(' ', '_')
    with open(file_path, 'w', encoding="utf8") as file:
        csv_writer = csv.DictWriter(file,
                                    quotechar='"',
                                    fieldnames=['text', 'sentiment'])
        csv_writer.writeheader()
        for tweet in tweets:
            csv_writer.writerow(tweet)
    return file_path


if __name__ == '__main__':
    filePath = dump_tweets(query="donald trump", count=10)  #
    tweets_sentiment = pd.read_csv(filePath).to_dict('records')
    print("Found %d tweets" % len(tweets_sentiment))
    positive_tweets = [tweet['text'] for tweet
                       in tweets_sentiment if tweet['sentiment'] == 'positive']
    negative_tweets = [tweet['text'] for tweet
                       in tweets_sentiment if tweet['sentiment'] == 'negative']
    neutral_tweets = [tweet['text'] for tweet
                      in tweets_sentiment if tweet['sentiment'] == 'neutral']
    print("Positive Tweets %d" % len(positive_tweets))
    topic_modeling(tweets=positive_tweets)
    print("Negative Tweets %d" % len(negative_tweets))
    topic_modeling(tweets=negative_tweets)
    print("Neutral Tweets %d" % len(neutral_tweets))
    topic_modeling(tweets=neutral_tweets)
