from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
import requests
import zipfile


class SRDataset(Dataset):

    def __init__(self, data_downloader, split):
        self.nodes_data, self.y_true_dict, train_contexts, val_contexts, test_contexts = data_downloader.get_data()

        self.contexts = {
            "train": train_contexts,
            "val": val_contexts,
            "test": test_contexts,
            "testdata": train_contexts + val_contexts + test_contexts
        }[split]

    def __len__(self):

        return len(self.contexts)

    def __getitem__(self, idx):
        context_idx = self.contexts[idx]

        context_features = self.nodes_data[context_idx]
        labels = self.y_true_dict[context_idx]

        return context_idx, context_features, labels


class SRProcessedDataset(Dataset):

    def __init__(self, context_indices, features, labels, label_lengths, node_batch_size, node_shuffle):
        self.context_data_loaders = []

        for context_features, context_labels, context_label_lengths in zip(features, labels, label_lengths):
            cds = DataLoader(ContextDataset(context_features, context_labels, context_label_lengths), shuffle=node_shuffle, batch_size=node_batch_size)
            self.context_data_loaders.append(cds)

        self.context_indices = context_indices

    def __len__(self):

        return len(self.context_indices)

    def __getitem__(self, idx):

        return self.context_indices[idx], self.context_data_loaders[idx]


class ContextDataset(Dataset):

    def __init__(self, features, labels, label_lengths):
        self.features = features
        self.labels = labels
        self.label_lengths = label_lengths

    def __getitem__(self, idx):

        return self.features[idx], self.labels[idx], self.label_lengths[idx]

    def __len__(self):

        return len(self.labels)


class DataDownloader:

    def __init__(self, scenario=1, use_cache=True):
        self.scenario = scenario
        self.data_path = "data"
        self.outputs_path = f"{self.data_path}/output_11ax_sr_simulations_sce{scenario}.txt"
        self.inputs_path = f"{self.data_path}/simulator_input_files_sce{scenario}"
        self.use_cache = use_cache

        self.cache_path = f"tmp/data_scenario{scenario}.pkl"

        self.inputs_url = f"https://zenodo.org/record/5575120/files/simulator_input_files_sce{scenario}.zip?download=1"
        self.outputs_url = f"https://zenodo.org/record/5575120/files/output_11ax_sr_simulations_sce{scenario}.txt?download=1"

        if scenario == "test":
            self.outputs_path = self.outputs_path.replace(f"sce{scenario}", "test")
            self.inputs_path = self.inputs_path.replace(f"sce{scenario}", "test")
            self.inputs_url = self.inputs_url.replace(f"sce{scenario}", "test")
            self.outputs_url = self.outputs_url.replace(f"sce{scenario}", "test")

        self.download_data_if_not_exist(self.inputs_path, self.outputs_path, self.inputs_url, self.outputs_url)

        self.nodes, self.y_true_dict, self.train_contexts, self.val_contexts, self.test_contexts = self._load_nodes_cached() if self.use_cache else self._load_nodes()

    def download_data_if_not_exist(self, inputs_path, outputs_path, inputs_url, outputs_url):
        if not os.path.exists(outputs_path):
            print(f"Now downloading {outputs_path}...")
            r = requests.get(outputs_url, allow_redirects=True)

            open(outputs_path, 'wb').write(r.content)
            print(f"Downloaded {outputs_path}!")

        if not os.path.exists(inputs_path):
            print(f"Now downloading {inputs_path}...")
            r = requests.get(inputs_url, allow_redirects=True)

            zip_target = f"{inputs_path}.zip"

            open(zip_target, 'wb').write(r.content)

            with zipfile.ZipFile(zip_target, "r") as zip_ref:
                zip_ref.extractall(self.inputs_path if self.scenario == "test" else self.data_path)

            if self.scenario == 3:
                os.rename("data/input_files_3_light", "data/simulator_input_files_sce3")

            os.remove(zip_target)
            print(f"Downloaded {self.inputs_path}!")


    def get_data(self):

        return self.nodes, self.y_true_dict, self.train_contexts, self.val_contexts, self.test_contexts

    def _load_nodes_cached(self):
        # Check for cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        # Load input files
        res = self._load_nodes()

        # Cache contexts
        with open(self.cache_path, 'wb') as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

        return res

    def _load_nodes(self):
        inputs = self._load_inputs()

        f = open(self.outputs_path, "r", encoding="utf-8", errors="ignore")

        nodes = {}

        cur_simulation = {}
        cur_step = ["initial", "throughput", "interference", "rssi", "sinr"]
        for line in f:
            if line == "\n":
                break

            if cur_step[0] == "initial":
                cur_simulation = {}
                if self.scenario == 3:
                    cur_simulation["scenario"], cur_simulation["vidx"], cur_simulation["threshold"], _ = re.findall('\d+', line)
                else:
                    cur_simulation["scenario"], cur_simulation["threshold"], _ = re.findall('\d+', line)

                cur_simulation["threshold"] = "-" + cur_simulation["threshold"]
            else:
                nums = re.findall('(-?\d+\.\d+|-nan|-inf|nan|inf)', line)
                nums = [num.replace("nan", "0").replace("inf", "0") for num in nums]
                cur_simulation[cur_step[0]] = list(map(float, nums))

            cur_step = cur_step[1:] + cur_step[:1]  # turn step

            if cur_step[0] == "initial":
                if not cur_simulation["scenario"] in nodes:
                    nodes[cur_simulation["scenario"]] = {}

                csv_name = f"input_nodes_s{cur_simulation['scenario']}_{'_v'+cur_simulation['vidx']+'_' if self.scenario == 3 else ''}c{cur_simulation['threshold']}.csv"
                #cur_simulation["input_nodes"] = inputs[csv_name] # TODO: include

                nodes[cur_simulation["scenario"]][cur_simulation["threshold"]] = cur_simulation

        y_true_dict = self._calculate_y_true_dict(nodes)

        train_contexts, test_contexts = train_test_split(list(y_true_dict.keys()), test_size=0.20, random_state=1, shuffle=True)
        val_contexts, test_contexts = train_test_split(test_contexts, test_size=0.50, random_state=1, shuffle=True)

        return nodes, y_true_dict, train_contexts, val_contexts, test_contexts

    def _calculate_y_true_dict(self, nodes):

        y_true_dict = {sim: {} for sim in nodes.keys()}

        for sim in nodes.keys():
            for threshold in nodes[sim].keys():
                y_true_dict[sim][threshold] = nodes[sim][threshold]["throughput"]

        return y_true_dict

    def _load_inputs(self):
        inputs = {}
        for fname in os.listdir(self.inputs_path):
            if fname.endswith(".csv"):
                fpath = os.path.join(self.inputs_path, fname)

                df = pd.read_csv(fpath, delimiter=";")

                inputs[fname] = df.to_dict()

        return inputs
