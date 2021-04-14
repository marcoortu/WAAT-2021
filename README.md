# WAAT-2021

Social Mining

## Configurazione credenziali di accesso alle WEB API

Creare il file credentials.py nel quale inserire le credenziali 
dei vari social media. 

```python
# Mettere qui i dati corretti per Twitter
TWITTER_CONSUMER_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXX'
TWITTER_CONSUMER_SECRET = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
TWITTER_ACCESS_TOKEN = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
TWITTER_ACCESS_TOKEN_SECRET = 'XXXXXXXXXXXXXXXXXXXXXXXXX'

FACEBOOK_ACCESS_TOKEN = 'XXXXXXXXXXXXXXXXXXXXXXXX'
```

## Twitter Scraper

Se non si ha accesso alle API di Twitter è possibile utilizzare una delle tante 
librerie di _scraping_ disponibili ad esempio: [SNScrape](https://github.com/JustAnotherArchivist/snscrape). 
Un esempio di utilizzo si trova nel file _twitter_scraper_examples.py_.

## Facebook Scraper

Se non si ha accesso alle API di Facebook è possibile utilizzare una delle tante 
librerie di _scraping_ disponibili ad esempio: [facebook-scraper](https://pypi.org/project/facebook-scraper/). 
Un esempio di utilizzo si trova nel file _facebook_scraper_exaples.py_.

## Twitter


### Esercizio 1

Estrarre 1000 tweets con con un dato criterio di ricerca e utilizzare la topic modeling analysis
(Latent Dirichlet Allocation per esempio) per ottenere i 5 topic più popolari.

1. Estrarre i tweets e salvarli in un csv.
2. Utilizzando la sentiment analysis dividere i tweet in positivi, negativi e neutrali.
3. Per ciascun gruppo utilizzare la LDA per estrarre i topics.


## LinkedIn

### Esercizio 1

Clusterizzare i colleghi delle propria rete LinkedIn in base alla "job position". 


1. Utilizzare il file csv delle connections scaricato da LinkedIn.
2. Calcolare la matrice TF-IDF utilizzando il TF-IDF vectorizer.
3. Utilizzare il KMeans per clusterizzare le "Job Positions".
4. Utilizzando il metodo "Elbow" ottenere una stima del k migliore.
5. In base alla vostra "Job Positions" calcolate il cluster a cui appartenete. 
6. Calcolare la distribuzione di campioni per cluster utilizzando PrettyTable per visualizzare i dati.
7. Utilizzare una decomposizione PCA o MDS per visualizzare i clusters ottenuti:
    ```python
    from sklearn.manifold import MDS
    from sklearn.decomposition import PCA
    # USE MDS 
    decomposition = MDS(n_components=2,
                  dissimilarity="precomputed",
                  random_state=1)
    # OR PCA
    decomposition = PCA(n_components=2)
    ```


