import os
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
import requests
import zipfile

class DataLoader:

    def __init__(self, scenario=1, use_cache=True):
        self.data_path = "data"
        self.outputs_path = f"{self.data_path}/output_11ax_sr_simulations_sce{scenario}.txt"
        self.inputs_path = f"{self.data_path}/simulator_input_files_sce{scenario}"
        self.use_cache = use_cache
        
        self.cache_path = f"tmp/data_scenario{scenario}.pkl"

        self.inputs_url = f"https://zenodo.org/record/5506248/files/simulator_input_files_sce{scenario}.zip?download=1"
        self.outputs_url = f"https://zenodo.org/record/5506248/files/output_11ax_sr_simulations_sce{scenario}.txt?download=1"

        self.download_data_if_not_exist()

        self.nodes, self.y_true_dict, self.train_contexts, self.test_contexts = self._load_nodes_cached() if self.use_cache else self._load_nodes()

    def download_data_if_not_exist(self):
        if not os.path.exists(self.outputs_path):
            print(f"Now downloading {self.outputs_path}...")
            r = requests.get(self.outputs_url, allow_redirects=True)

            open(self.outputs_path, 'wb').write(r.content)
            print(f"Downloaded {self.outputs_path}!")

        if not os.path.exists(self.inputs_path):
            print(f"Now downloading {self.inputs_path}...")
            r = requests.get(self.inputs_url, allow_redirects=True)

            zip_target = f"{self.inputs_path}.zip"

            open(zip_target, 'wb').write(r.content)

            with zipfile.ZipFile(zip_target, "r") as zip_ref:
                zip_ref.extractall(self.data_path)

            os.remove(zip_target)
            print(f"Downloaded {self.inputs_path}!")

    def get_data(self):

        return self.nodes, self.y_true_dict, self.train_contexts, self.test_contexts

    def _load_nodes_cached(self):
        # Check for cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        # Load input files
        nodes, y_true_dict, train_contexts, test_contexts = self._load_nodes()

        # Cache contexts
        with open(self.cache_path, 'wb') as f:
            pickle.dump((nodes, y_true_dict, train_contexts, test_contexts), f, protocol=pickle.HIGHEST_PROTOCOL)

        return nodes, y_true_dict, train_contexts, test_contexts

    def _load_nodes(self):
        inputs = self._load_inputs()

        f = open(self.outputs_path, "r")

        nodes = {}

        cur_simulation = {}
        cur_step = ["initial", "throughput", "interference", "rssi", "sinr"]
        for line in f:
            if line == "\n":
                break

            if cur_step[0] == "initial":
                cur_simulation = {}
                cur_simulation["scenario"], cur_simulation["threshold"], _ = re.findall('\d+', line)
                cur_simulation["threshold"] = "-" + cur_simulation["threshold"]
            else:
                line = line.strip("\\n")
                line = line.replace("f48.29", "48.29") # Fixes the bug in scenario 1.
                cur_simulation[cur_step[0]] = list(map(float, line.split(",")))

            cur_step = cur_step[1:] + cur_step[:1] # turn step

            if cur_step[0] == "initial":
                if not cur_simulation["scenario"] in nodes:
                    nodes[cur_simulation["scenario"]] = {}

                cur_simulation["input_nodes"] = inputs[f"input_nodes_s{cur_simulation['scenario']}_c{cur_simulation['threshold']}.csv"]

                nodes[cur_simulation["scenario"]][cur_simulation["threshold"]] = cur_simulation

        y_true_dict = self._calculate_y_true_dict(nodes)

        train_contexts, test_contexts = train_test_split(list(y_true_dict.keys()), test_size=0.2, random_state=1, shuffle=True)

        return nodes, y_true_dict, train_contexts, test_contexts

    def _calculate_y_true_dict(self, nodes):

        y_true_dict = {sim: {} for sim in nodes.keys()}

        for sim in nodes.keys():
            for threshold in nodes[sim].keys():
                y_true_dict[sim][threshold] = nodes[sim][threshold]["throughput"][0]
        
        return y_true_dict
    
    def _load_inputs(self):
        inputs = {}
        for fname in os.listdir(self.inputs_path):
            if fname.endswith(".csv"):
                fpath = os.path.join(self.inputs_path, fname)

                df = pd.read_csv(fpath)

                inputs[fname] = df

        return inputs


if __name__ == "__main__":
    # Test data loading
    r = DataLoader()

    nodes, y_true_dict, train_contexts, test_contexts = r.get_data()

    print(nodes["0000"]["-68"]["throughput"])
    assert nodes["0000"]["-68"]["throughput"] == [17.0]
