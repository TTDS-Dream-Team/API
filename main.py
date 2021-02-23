from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from pymongo import MongoClient
import zlib
import time
from typing import *
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import OrderedDict
from operator import getitem
from lsh import LSH
from nn import NN

app = FastAPI(title="BetterReads API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("paraphrase-distilroberta-base-v1")
model_sentiment = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

data_file = os.getenv('DATA_FILE')
db_sentences = os.getenv('DB_SENTENCES')

if data_file is None:
    lsh = LSH()
else:
    lsh = LSH(hdf5_file=data_file)

nn = NN()

db_pwd = os.getenv('MONGO_PWD')
client = MongoClient(f'mongodb+srv://cdminix:{db_pwd}@cluster0.pdjrf.mongodb.net/Reviews_Data?retryWrites=true&w=majority')
db = client.Reviews_Data


def get_rating(sentiment):
    """Returns (rating, confidence) tuple from transformer sentiment output"""
    rating_str = sentiment[0]['label']
    rating = int(rating_str.split(" ")[0])
    confidence = float(sentiment[0]['score'])
    return (rating, confidence)


def rank_by_sentiment(data, query_rating):
    for review_id, values in data.items():
        review_rating = values.get('rating')
        print(review_rating)
        sentiment_similarity = 5 - abs(query_rating - review_rating)
        values['sentiment_similarity'] = sentiment_similarity

    # Rank data by sentiment value
    sentiment_order = OrderedDict(
        sorted(data.items(), key=lambda x: getitem(x[1], 'sentiment_similarity'), reverse=True))
    return dict(sentiment_order)


def elapsed_time():
    e_time = time.time()
    if not hasattr(elapsed_time, 's_time'):
        elapsed_time.s_time = e_time
    else:
        time_diff = round(1000 * (e_time - elapsed_time.s_time), 2)
        elapsed_time.s_time = e_time
        return time_diff
    return None


@app.get("/search/{query}")
def get_query(query: str, measure_time: Optional[bool] = False):
    if measure_time:
        elapsed_time()
        time_dict = {}
    
    # encode the text using the transformer model
    query = model.encode([query])[0]

    # Get query sentiment
    query_rating = get_rating(model_sentiment(query))[0]
    
    if measure_time:
        time_dict['encoding'] = elapsed_time()
    
    # get the bucket the vector is in
    hashed, vectors = lsh.get(query)
    #vectors = np.matrix(vectors)
    
    if measure_time:
        time_dict['get_bucket'] = elapsed_time()
    
    # get the nearest neighbors in said bucket
    ids = [f"{hashed}_{int(id)}" for id in nn.get_k_nn(lsh.quantize(query), vectors, chunks=True)]
    
    print('neighbors', len(ids))

    if measure_time:
        time_dict['nn_search'] = elapsed_time()
    
    # get details from db
    sents = list(db[db_sentences].find({"_id": {"$in": ids}}))
    review_ids = [s['review'] for s in sents]
    reviews = list(db['reviews'].find({"_id": {"$in": review_ids}}))

    print('sents', len(sents))

    print('uids', set(ids))
    print('revids', review_ids)

    print('reviews', len(reviews))

    if measure_time:
        time_dict['db_calls'] = elapsed_time()

    for i, r in enumerate(reviews):
        reviews[i]['review_text'] = zlib.decompress(r['review_text'])
        for s in sents:
            if s['review'] == reviews[i]['_id']:
                 reviews[i]['relevant_text'] = reviews[i]['review_text'][s['start']:s['end']]
        reviews[i]['description'] = zlib.decompress(r['description'])
        del reviews[i]['_id']

    results = { i: r for i, r in enumerate(reviews)}

    if measure_time:
        time_dict['finalize_and_decompress'] = elapsed_time()
        time_dict['total'] = sum(time_dict.values())
        results['timings'] = time_dict

    # Rank results by sentiment
    sentiment_results = rank_by_sentiment(results, query_rating)
        
    return sentiment_results
