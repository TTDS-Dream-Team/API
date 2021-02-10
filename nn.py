import numpy as np
from tqdm.auto import tqdm

class NN():

    # Returns Euclidean distance between vectors
    def _distance_(self, v1, v2):
        return np.linalg.norm(v1 - v2)

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
                    index[max_idx] = chunk + i
                    dists[max_idx] = dist
                    max_dist = np.max(dists)

        return index[~np.isnan(index)]
