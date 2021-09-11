import os
import pandas as pd
import pickle

class Data:

    def __init__(self, data_path = "data/simulator_input_files", cache_path="tmp/contexts.pkl"):
        self.data_path = data_path
        self.cache_path = cache_path

        self.contexts = self.load()

    def load(self):
        # Check for cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(open(self.cache_path, 'rb'))

        # Load contexts
        contexts = {}
        for fname in os.listdir(self.data_path):
            if fname.endswith(".csv"):
                fpath = os.path.join(self.data_path, fname)

                fcsv = pd.read_csv(fpath)

                contexts[fname] = fcsv

        # Cache contexts
        with open(self.cache_path, 'wb') as f:
            pickle.dump(contexts, f, protocol=pickle.HIGHEST_PROTOCOL)

        return contexts


if __name__ == "__main__":
    # Test data loading
    r = Data()

    r.load()
    print(r.contexts)
