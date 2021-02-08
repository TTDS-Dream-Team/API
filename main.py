from fastapi import FastAPI
import numpy as np
import h5py
import os
from pymongo import MongoClient
import zlib
import time
from typing import *

app = FastAPI(title="BetterReads API")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-distilroberta-base-v1")


class LSH:
    def __init__(
        self,
        hdf5_file="../data.h5py",
        input_dim=768,
        hash_dim=6,
        seed=42,
        chunksize=1_000,
        dtype="int8",
    ):
        self.planes = []
        self.input_dim = input_dim
        np.random.seed(seed)
        for i in range(hash_dim):
            v = np.random.rand(input_dim)
            v_hat = v / np.linalg.norm(v)
            self.planes.append(v_hat)

        self.planes = np.matrix(self.planes)
        self.data = h5py.File(hdf5_file, "r")
        self.buckets = {}
        self.dtype = dtype

    # Returns LSH of a vector
    def hash(self, vector):
        hash_vector = np.where((self.planes @ vector) < 0, 1, 0)[0]
        hash_string = "".join([str(num) for num in hash_vector])
        return hash_string

    def quantize(self, vector_list):
        vector_list = np.array(vector_list)
        if self.dtype in ["float16", "float32"]:
            return vector_list.astype(self.dtype)
        if self.dtype == "int8":
            return np.asarray(vector_list * 128, dtype=np.int8)
        raise ValueError(f"dtype needs to be float32, float16 or int8")

    # Returns bucket vector is in
    def get(self, vector):
        hashed = self.hash(vector)

        if hashed in self.data:
            return hashed, self.data[hashed]

        return hashed, []


lsh = LSH()


class NN:

    # Returns Euclidean distance between vectors
    def _distance_(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    # Returns v1's n Nearest Neighbours in vectors
    def get_nn(self, v1, vectors, n=10):
        dists = np.repeat(np.inf, n)
        index = np.repeat(np.nan, n)

        max_dist = np.inf

        for i, v2 in enumerate(vectors):
            dist = self._distance_(v1, v2)
            if dist < max_dist:
                max_idx = np.argmax(dists)
                index[max_idx] = i
                dists[max_idx] = dist
                max_dist = np.max(dists)

        return index[~np.isnan(index)]


nn = NN()

db_pwd = os.getenv('MONGO_PWD')
client = MongoClient(f'mongodb+srv://cdminix:{db_pwd}@cluster0.pdjrf.mongodb.net/Reviews_Data?retryWrites=true&w=majority')
db = client.Reviews_Data

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
    query = model.encode([query])[0]
    if measure_time:
        time_dict['encoding'] = elapsed_time()
    hashed, vectors = lsh.get(query)
    if measure_time:
        time_dict['get_bucket'] = elapsed_time()
    ids = [f"{hashed}_{int(id)}" for id in nn.get_nn(lsh.quantize(query), vectors)]
    if measure_time:
        time_dict['nn_search'] = elapsed_time()
    sents = list(db['sentence_data'].find({"_id": {"$in": ids}}))
    review_ids = [s['review'] for s in sents]
    reviews = list(db['review_data'].find({"_id": {"$in": review_ids}}))
    isbns = [r['isbn'] for r in reviews]
    books = list(db['book_data'].find({"isbn": {"$in": isbns}}))
    results = {}
    time_dict['db_calls'] = elapsed_time()
    for s in sents:
        review = [r for r in reviews if s['review'] == r['_id']][0]
        book = [b for b in books if b['isbn'] == review['isbn']][0]
        text = zlib.decompress(review['review_text'])
        results[ids.index(s['_id'])] = {
            'rank': ids.index(s['_id']),
            'text': text,
            'relevant_text': text[s['start']:s['end']],
            'relevant_range': [s['start'], s['end']],
            'isbn': book['isbn'],
            'image': book['image_url'],
            'title': book['title'],
        }
    time_dict['finalize_and_decompress'] = elapsed_time()
    time_dict['total'] = sum(time_dict.values())
    results['timings'] = time_dict
    return results
