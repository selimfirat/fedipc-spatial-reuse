import os
import pandas as pd
from tqdm import tqdm

inputs_paths = ["data/simulator_input_files_sce1", "data/simulator_input_files_sce2"]


res = set()
for inputs_path in inputs_paths:
    for fname in tqdm(os.listdir(inputs_path)):
        if fname.endswith(".csv"):
            fpath = os.path.join(inputs_path, fname)

            df = pd.read_csv(fpath, delimiter=";")
            res = res.union(set(df.columns[df.nunique() > 1].tolist()))

    print(inputs_path)
    print(";".join(res))
