import os
import pandas as pd
import pickle
import re


class DataLoader:

    def __init__(self, inputs_path = "data/simulator_input_files", sim_outputs_path="data/output_11ax_sr_simulations.txt", use_cache=True, cache_path="tmp/nodes.pkl"):
        self.sim_outputs_path = sim_outputs_path
        self.use_cache = use_cache
        self.data_path = inputs_path
        self.cache_path = cache_path

        self.nodes, self.y_true_dict = self._load_nodes_cached() if self.use_cache else self._load_nodes()

    def get_data(self):

        return (self.nodes, self.y_true_dict)

    def _load_nodes_cached(self):
        # Check for cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        # Load input files
        nodes, y_true_dict = self._load_nodes()

        # Cache contexts
        with open(self.cache_path, 'wb') as f:
            pickle.dump((nodes, y_true_dict), f, protocol=pickle.HIGHEST_PROTOCOL)

        return nodes, y_true_dict

    def _load_nodes(self):
        inputs = self._load_inputs()

        f = open(self.sim_outputs_path, "r")

        nodes = {}

        cur_simulation = {}
        cur_step = ["initial", "throughput", "interference", "rssi", "sinr"]
        for line in f:
            if line == "\n":
                break

            if cur_step[0] == "initial":
                cur_simulation["scenario"], cur_simulation["threshold"], _ = re.findall('\d+', line)
                cur_simulation["threshold"] = "-" + cur_simulation["threshold"]
            else:
                cur_simulation[cur_step[0]] = list(map(float, line.split(",")))

            cur_step = cur_step[1:] + cur_step[:1] # turn step

            if cur_step[0] == "initial":
                if not cur_simulation["scenario"] in nodes:
                    nodes[cur_simulation["scenario"]] = {}

                cur_simulation["input_nodes"] = inputs[f"input_nodes_s{cur_simulation['scenario']}_c{cur_simulation['threshold']}.csv"]

                nodes[cur_simulation["scenario"]][cur_simulation["threshold"]] = cur_simulation

        y_true_dict = self._calculate_y_true_dict(nodes)

        return nodes, y_true_dict

    def _calculate_y_true_dict(self, nodes):

        y_true_dict = {sim: {} for sim in nodes.keys()}

        for sim in nodes.keys():
            for threshold in nodes[sim].keys():
                y_true_dict[sim][threshold] = nodes[sim][threshold]["throughput"][0]
        
        return y_true_dict
    
    def _load_inputs(self):
        inputs = {}
        for fname in os.listdir(self.data_path):
            if fname.endswith(".csv"):
                fpath = os.path.join(self.data_path, fname)

                df = pd.read_csv(fpath)

                inputs[fname] = df

        return inputs


if __name__ == "__main__":
    # Test data loading
    r = DataLoader()

    nodes, y_true_dict = r.get_data()