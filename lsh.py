import numpy as np
import h5py

class LSH:
    def __init__(self,
                 hdf5_file="data.hdf5",
                 input_dim=768,
                 hash_dim=6,
                 seed=42,
                 chunksize=1_000,
                 dtype='int8',
                 file_write='w',
                 add_neg=True,
                 random_factor=3,
        ):
      self.planes = []
      self.input_dim = input_dim
      np.random.seed(seed)
      factor = random_factor
      for i in range(hash_dim*factor):
          if add_neg:
              v = (np.random.rand(input_dim) * 2) - 1
          else:
              v = np.random.rand(input_dim)
          v_hat = v / np.linalg.norm(v)
          self.planes.append(v_hat)
      dists = np.zeros((hash_dim*factor,hash_dim*factor))
      for i in range(hash_dim*factor):
          for j in range(hash_dim*factor):
                if i == j:
                    dists[i,j] = np.inf
                else:
                    dists[i,j] = np.linalg.norm(self.planes[i]-self.planes[j])
      remove_idx = []
      while len(remove_idx) < hash_dim*factor - hash_dim:
          ind = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
          if ind[0] not in remove_idx:
              remove_idx.append(ind[0])
          dists[ind] = np.inf
          dists[ind[1],ind[0]] = np.inf
      print(remove_idx, len(remove_idx), len(set(remove_idx)), hash_dim)
      new_planes = []
      for i, p in enumerate(self.planes):
          if i not in remove_idx:
                new_planes.append(self.planes[i])
      print(len(new_planes), len(self.planes))
  
      self.planes = np.matrix(new_planes)
      self.data = h5py.File(hdf5_file, file_write)
      self.chunksize = chunksize
      self.buckets = {}
      self.dtype = dtype
    
    # Returns LSH of a vector
    def hash(self, vector):
      hash_vector = np.where((self.planes @ vector) < 0, 1, 0)[0]
      hash_string = "".join([str(num) for num in hash_vector])
      return hash_string
    
    def quantize(self, item_list):
      vector_list = [i['vector'] for i in item_list]
      vector_list = np.array(vector_list)
      if self.dtype in ['float16', 'float32']:
          return vector_list.astype(self.dtype)
      if self.dtype == 'int8':
          return np.asarray(vector_list * 128, dtype=np.int8)
      raise ValueError(f'dtype needs to be float32, float16 or int8')
    
    def dict_to_hdf5(self, hashed, flush=True):
      list_size = self.chunksize
      if flush:
        list_size = len(self.buckets[hashed])
      if len(self.buckets[hashed]) >= list_size and list_size > 0:
          items = self.buckets[hashed]
          if hashed not in self.data:
              self.data.create_dataset(
                  hashed,
                  (list_size,self.input_dim),
                  compression='gzip',
                  dtype=self.dtype,
                  chunks=True,
                  #chunks=(10_000,self.input_dim),
                  maxshape=(None,self.input_dim)
              )
          else:
              hf = self.data[hashed]
              hf.resize((hf.shape[0] + list_size), axis=0)
          self.data[hashed][-list_size:] = self.quantize(self.buckets[hashed])
          self.buckets[hashed] = []
          idx = np.arange(list_size) + len(self.data[hashed]) - 1
          for i, id in enumerate(idx):
            del items[i]['vector']
            items[i]['_id'] = f'{hashed}_{id}'
          return items
      return []

    # Add vector to bucket
    def add(self, item):
      vector = item['vector']
      hashed = self.hash(vector)
      
      if hashed not in self.buckets:
          self.buckets[hashed] = []
      
      self.buckets[hashed].append(item)
      
      return self.dict_to_hdf5(hashed)
            
    def flush(self):
      items = []
      for hashed in self.buckets.keys():
        items += self.dict_to_hdf5(hashed, flush=True)
      return items
    
    # Returns bucket vector is in
    def get(self, vector):
      hashed = self.hash(vector)
      if hashed in self.data:
          return self.data[hashed]
      return []
