from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
import requests
import zipfile


class SRDataset(Dataset):

    def __init__(self, data_downloader, is_train):
        self.is_train = is_train
        self.nodes_data, self.y_true_dict, train_contexts, test_contexts = data_downloader.get_data()

        self.contexts = train_contexts if self.is_train else test_contexts

    def __len__(self):

        return len(self.contexts)

    def __getitem__(self, idx):
        context_idx = self.contexts[idx]

        context_features = self.nodes_data[context_idx]
        labels = self.y_true_dict[context_idx]

        return context_idx, context_features, labels


class SRProcessedDataset(Dataset):

    def __init__(self, context_indices, features, labels, node_batch_size, node_shuffle):
        self.context_data_loaders = []

        for context_features, context_labels in zip(features, labels):
            cds = DataLoader(ContextDataset(context_features, context_labels), shuffle=node_shuffle, batch_size=node_batch_size)
            self.context_data_loaders.append(cds)

        self.context_indices = context_indices

    def __len__(self):

        return len(self.context_indices)

    def __getitem__(self, idx):

        return self.context_indices[idx], self.context_data_loaders[idx]


class ContextDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):

        return self.features[idx], self.labels[idx]

    def __len__(self):

        return len(self.labels)


class DataDownloader:

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

        f = open(self.outputs_path, "r", encoding="utf-8")

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
                nums = re.findall('(-?\d+\.\d+|-nan|-inf|nan|inf)', line)
                nums = [num.replace("nan", "0").replace("inf", "0") for num in nums]  # TODO: Fix nan issue
                cur_simulation[cur_step[0]] = list(map(float, nums))

            cur_step = cur_step[1:] + cur_step[:1]  # turn step

            if cur_step[0] == "initial":
                if not cur_simulation["scenario"] in nodes:
                    nodes[cur_simulation["scenario"]] = {}

                # cur_simulation["input_nodes"] = inputs[f"input_nodes_s{cur_simulation['scenario']}_c{cur_simulation['threshold']}.csv"] # TODO: include

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
