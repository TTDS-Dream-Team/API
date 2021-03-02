import numpy as np
from tqdm.auto import tqdm

class NN():

    # Returns Euclidean distance between vectors
    def _distance_(self, v1, v2):
        return np.linalg.norm(v1 - v2)
    
    # Returns Euclidean distance between vector and matrix of vectors
    def _matrix_distance_(self, v1, matrix):
        # Conversion to float prevents overflow from following operations
        v1 = np.array([v1]).astype("float")
        matrix = matrix.astype("float")
        
        p1 = np.sum(v1**2, axis=1)[:, np.newaxis]
        p2 = np.sum(matrix**2, axis=1)
        p3 =  -2 * np.dot(v1, matrix.T)

        return p1 + p2 + p3

    # Returns v1's n Nearest Neighbours in vectors
    def get_nn(self, v1, vectors, n=10):
        # load all vectors into memory
        print(f'searching {len(vectors)} vectors')

        dists = np.repeat(np.inf, n)
        index = np.repeat(np.nan, n)

        max_dist = np.inf

        chunk_size = 10_000

        for chunk in tqdm(range(len(vectors)//chunk_size)):
            for i, v2 in enumerate(vectors[chunk:chunk+chunk_size]):
                dist = self._distance_(v1, v2)
                if dist < max_dist:
                    max_idx = np.argmax(dists)
                    if chunk + i not in index:
                        index[max_idx] = chunk + i
                        dists[max_idx] = dist
                        max_dist = np.max(dists)

        print(dists)
        return index[~np.isnan(index)]

    # Updated kNN for matrix ops
    def get_k_nn(self, v1, vectors, k=100, chunks=False):
        print(f'searching {len(vectors)} vectors')

        print(type(vectors))

        if chunks:
            dists = np.empty((len(vectors),3))
            chunksize = 10_000
            for c in vectors.iter_chunks():
                dists[c[0],0] = self._matrix_distance_(v1, vectors[c])
        else:
            vectors = np.array(vectors)
            dists = self._matrix_distance_(v1, vectors)
        
        if chunks:
            top_k = np.argsort(dists[:,0])[:k]
            dists = dists[top_k]
        else:
            top_k = np.argsort(dists, axis=1)[0,:k]
        
        #results = vectors[top_k[:],:]
        results = top_k

        return results, dists

    # Calculate Levenstein distance between v1 and all vectors
    def _levenstein_distance_(self, v1, vectors):
        word1 = v1
        scores = []
        for i in range(len(vectors)):
            score = 0
            word2 = vectors[i]
            
            for j in range(max(len(word1), len(word2))):
                char1 = -1 if j >= len(word1) else ord(word1[j])
                char2 = -1 if j >= len(word2) else ord(word2[j])

                diff = char1 - char2

                if diff != 0:
                    score += 1
            scores.append(score)
        return np.array(scores)

    # Sort results by Levenstein distance
    # May not be needed
    def sort_by_levenstein_distance(self, v1, vectors):
        distances = self._levenstein_distance_(v1, vectors)

        sorted_idx = np.argsort(distances)
        return vectors[sorted_idx[:],:]


