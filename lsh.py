import numpy as np
import h5py

class LSH():
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