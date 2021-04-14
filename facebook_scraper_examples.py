from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from facebook_scraper import get_posts
from prettytable import PrettyTable
from textblob import TextBlob


def get_posts_data(query, count=100, pages=3):
    posts = get_posts(query, pages=pages)
    post_list = []
    for post in posts:
        if len(post_list) > count:
            break
        try:
            if 'text' in post and len(post_list) < count:
                post_list.append(post)
        except KeyError:
            pass
    return post_list


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def comparative_sentiment(
        entity1="CocaColaUnitedStates",
        entity2="PepsiMaxUSA",
        count=100,
        pages=5):
    # recupero il testo dei posts
    entity1_posts = get_posts_data(entity1, count=count, pages=pages)
    entity2_posts = get_posts_data(entity2, count=count, pages=pages)
    print(entity2_posts[0])
    # estraggo solo il testo del post
    entity1_messages = [post['text'] for post in entity1_posts]
    entity2_messages = [post['text'] for post in entity2_posts]
    # calcolo il sentiment per i singoli messaggi
    entity1_sentiment = [get_sentiment(message) for message in entity1_messages]
    entity2_sentiment = [get_sentiment(message) for message in entity2_messages]
    print(entity1_sentiment)
    # imposto la tabella
    pretty_table = PrettyTable(field_names=['Metric', entity1, entity2])
    # numero di posts
    pretty_table.add_row([
        '# Posts',
        len(entity1_sentiment),
        len(entity2_sentiment)
    ])
    # sentiment medio
    pretty_table.add_row([
        'Sentiment AVG',
        '%0.2f' % np.mean(entity1_sentiment),
        '%0.2f' % np.mean(entity2_sentiment)
    ])
    pretty_table.add_row([
        'Sentiment AVG No Neutral',
        "%.2f" % np.mean(list(filter(lambda x: x != 0.0, entity1_sentiment))),
        "%.2f" % np.mean(list(filter(lambda x: x != 0.0, entity2_sentiment)))
    ])
    # calcolo la percentuale di commenti neutrali/positivi/negativi filtrando i valori dalla lista dei sentiment
    pretty_table.add_row([
        'Neutral',
        '%0.2f%%' % (len(list(filter(lambda x: x == 0.0, entity1_sentiment))) / count * 100),
        '%0.2f%%' % (len(list(filter(lambda x: x == 0.0, entity2_sentiment))) / count * 100)
    ])
    pretty_table.add_row([
        'Positive',
        '%0.2f%%' % (len(list(filter(lambda x: x > 0.0, entity1_sentiment))) / count * 100),
        '%0.2f%%' % (len(list(filter(lambda x: x > 0.0, entity2_sentiment))) / count * 100)
    ])
    pretty_table.add_row([
        'Negative',
        '%0.2f%%' % (len(list(filter(lambda x: x < 0.0, entity1_sentiment))) / count * 100),
        '%0.2f%%' % (len(list(filter(lambda x: x < 0.0, entity2_sentiment))) / count * 100)
    ])
    pretty_table.align['Metric'] = 'l'
    print(pretty_table)
    # Faccio il boxplot dei risultati
    labels = [entity1, entity2]
    plt.figure()
    plt.boxplot(
        [entity1_sentiment, entity2_sentiment],  # passo le due serie di dati
        labels=labels
    )
    plt.show()
    entity1_dates = [
        post['time'].strftime('%Y-%m-%d')
        for post in entity1_posts
    ]
    entity2_dates = [
        post['time'].strftime('%Y-%m-%d')
        for post in entity2_posts
    ]
    plt.figure()
    plt.plot(entity1_dates, entity1_sentiment)
    plt.plot(entity2_dates, entity2_sentiment)
    plt.legend([entity1, entity2])
    plt.show()


def comparative_sentiment_over_time(
        entity1="donaldtrump",
        entity2="hillaryclinton",
        count=100,
        pages=5):
    # recupero i post nel periodo desiderato
    entity1_posts = get_posts_data(entity1, count=count, pages=pages)
    entity2_posts = get_posts_data(entity2, count=count, pages=pages)
    # estraggo solo il testo del post
    entity1_messages = [post['text'] for post in entity1_posts]
    entity2_messages = [post['text'] for post in entity2_posts]
    # calcolo il sentiment per i singoli messaggi
    entity1_sentiment = [get_sentiment(message) for message in entity1_messages]
    entity2_sentiment = [get_sentiment(message) for message in entity2_messages]
    # Estraggo le date dai posts
    entity1_dates = [
        post['time'].strftime('%Y-%m-%d')
        for post in entity1_posts
    ]
    print(entity1_dates)
    entity2_dates = [
        post['time'].strftime('%Y-%m-%d')
        for post in entity2_posts
    ]
    # faccio il plot dei risultati
    plt.figure()
    plt.plot(entity1_dates, entity1_sentiment)
    plt.plot(entity2_dates, entity2_sentiment)
    plt.legend([entity1, entity2])
    plt.show()


def group_posts_sentiment_by_day(posts):
    messages_per_day = defaultdict(list)
    for post in posts:
        date_parsed = post['time'].strftime('%Y-%m-%d')
        sentiment = TextBlob(post['text']).sentiment.polarity
        messages_per_day[date_parsed].append(sentiment)
    return messages_per_day


def comparative_sentiment_avg_per_day(
        entity1="donaldtrump",
        entity2="biden",
        count=100,
        pages=5):
    # recupero i post nel periodo desiderato
    entity1_posts = get_posts_data(entity1, count, pages=pages)
    entity2_posts = get_posts_data(entity2, count, pages=pages)
    # Estraggo le date dai posts
    entity1_sentiment_per_day = group_posts_sentiment_by_day(entity1_posts)
    entity2_sentiment_per_day = group_posts_sentiment_by_day(entity2_posts)
    entity_dates = sorted(
        list(set(entity1_sentiment_per_day.keys()).intersection(set(entity2_sentiment_per_day.keys()))))
    entity1_sentiment = [np.mean(entity1_sentiment_per_day[date])
                         for date in entity_dates if date in entity1_sentiment_per_day]
    entity2_sentiment = [np.mean(entity2_sentiment_per_day[date])
                         for date in entity_dates if date in entity2_sentiment_per_day]

    plt.figure()
    plt.plot(entity_dates, entity1_sentiment)
    plt.plot(entity_dates, entity2_sentiment)
    plt.legend([entity1, entity2])
    plt.xticks(entity_dates, entity_dates, rotation='45')
    plt.show()


if __name__ == '__main__':
    comparative_sentiment(
        'donaldtrump',
        'joebiden',
        count=100,
        pages=20
    )
    comparative_sentiment_avg_per_day(
        'donaldtrump',
        'joebiden',
        count=100,
        pages=20
    )
