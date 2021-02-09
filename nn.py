import numpy as np

class NN():

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
