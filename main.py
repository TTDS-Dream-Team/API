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
import numpy as np
from lsh import LSH
from nn import NN
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')
cc = SmoothingFunction()

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
hyper_planes = int(os.getenv('HYPER_PLANES', 6))

if data_file is None:
    lsh = LSH()
else:
    lsh = LSH(hdf5_file=data_file, hash_dim=hyper_planes)

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


def sentiment_scores(reviews, query_rating):
    scores = np.zeros(len(reviews))
    for review_id, values in enumerate(reviews):
        review_rating = values.get('rating')
        #print(review_rating)
        if review_rating == 0:
            sentiment_similarity = 1
        else:
            sentiment_similarity = abs(query_rating - review_rating)
        scores[review_id] = sentiment_similarity

    return scores

def levenshtein(seq1, seq2, costs=[1, 1, 1]):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    ins_cost, eq_cost, del_cost = costs
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + ins_cost,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + del_cost
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + ins_cost,
                    matrix[x-1,y-1] + eq_cost,
                    matrix[x,y-1] + del_cost
                )
    return matrix[size_x - 1, size_y - 1]

def bleu(seq1, seq2):
    if len(seq1.lower().split()) == 0:
        return 1
    return 1-sentence_bleu([seq1.lower().split()], seq2.lower().split(), weights=[1,0,0,0], smoothing_function=cc.method0)

def meteor(seq, query):
    if len(query.lower().split()) == 0:
        return 1
    return 1 - meteor_score([seq], query)


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
    
    old_query = query

    # encode the text using the transformer model
    query = model.encode([query])[0]

    if measure_time:
        time_dict['encoding'] = elapsed_time()

    # Get query sentiment
    query_rating, query_confidence = get_rating(model_sentiment(old_query))
    
    if measure_time:
        time_dict['sentiment'] = elapsed_time()
    
    # get the bucket the vector is in
    hashed, vectors = lsh.get(query)
    #vectors = np.matrix(vectors)
    
    if measure_time:
        time_dict['get_bucket'] = elapsed_time()
    
    # get the nearest neighbors in said bucket
    nn_ids, dists = nn.get_k_nn(lsh.quantize(query), vectors, chunks=True)
    ids = [f"{hashed}_{int(id)}" for id in nn_ids]
    
    print('neighbors', len(ids))

    if measure_time:
        time_dict['nn_search'] = elapsed_time()
    
    # get details from db
    sents = list(db[db_sentences].find({"_id": {"$in": ids}}))
    review_ids = [s['review'] for s in sents]
    reviews = list(db['reviews'].find({"_id": {"$in": review_ids}}))

#    print('sents', len(sents))

#    print('uids', set(ids))
#    print('revids', review_ids)

#    print('reviews', len(reviews))

    if measure_time:
        time_dict['db_calls'] = elapsed_time()

    for i, r in enumerate(reviews):
        reviews[i]['review_text'] = zlib.decompress(r['review_text'])
        for s in sents:
            if s['review'] == reviews[i]['_id']:
                 reviews[i]['relevant_text'] = reviews[i]['review_text'][s['start']:s['end']]
        reviews[i]['description'] = zlib.decompress(r['description'])
        del reviews[i]['_id']

    if measure_time:
        time_dict['decompress'] = elapsed_time()

    dists = dists[:len(reviews)]
    for i, r in enumerate(reviews):
        #print(levenshtein(r['relevant_text'], old_query), old_query, r['relevant_text'])
        #dists[i,1] = levenshtein(r['relevant_text'], old_query)
        dists[i,1] = meteor(old_query, r['relevant_text'])
    print(dists[:,1])
    dists[:,2] = sentiment_scores(reviews, query_rating)
    dists = dists / dists.max(axis=0)
    if query_confidence < 0.5:
        query_confidence = 0
    weights = [0.9, 0.05, 0.05*query_confidence]
    for i, w in enumerate(weights):
        dists[:,i] = dists[:,i]*w
    sum_dists = dists.sum(axis=1)
    
    new_reviews = []
    for k in np.argsort(sum_dists)[:10]:
        reviews[k]['scores'] = {}
        reviews[k]['scores']['semantic'] = dists[k,0]
        reviews[k]['scores']['exact'] = dists[k,1]
        reviews[k]['scores']['rating'] = dists[k,2]
        new_reviews.append(reviews[k])

    reviews = new_reviews

    if measure_time:
        time_dict['sort_weight'] = elapsed_time()

    results = { i: r for i, r in enumerate(reviews)}

    results['sentiment'] = [query_rating, query_confidence]

    if measure_time:
        time_dict['finalize'] = elapsed_time()
        time_dict['total'] = sum(time_dict.values())
        results['timings'] = time_dict

    return results