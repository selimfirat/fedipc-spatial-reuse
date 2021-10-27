from dataset import SRDataset, DataDownloader
from torch.utils.data import DataLoader
import torch

from utils import seed_everything
import json

for scenario in [1, 2, 3]:
    seed_everything(1)

    print("Scenario", scenario)
    data_downloader = DataDownloader(scenario)

    train_data = SRDataset(data_downloader, split="train")
    train_loader = DataLoader(train_data)

    feature_names = ["throughput", "interference", "rssi", "sinr", "threshold"]

    result = {}
    for feature_name in feature_names:
        data = []
        for context_idx, features, labels in train_loader:
            for threshold, th_data in features.items():
                vals = th_data[feature_name]
                if feature_name == "threshold":
                    vals = [torch.Tensor([float(val)]) for val in vals]
                data.extend(torch.stack(vals))

        data = torch.stack(data)

        res = {
            "min": data.min().item(),
            "max": data.max().item(),
            "mean": data.mean().item(),
            "std": data.std().item()
        }

        result[feature_name] = res

    print(result)
"""
Sum of throughputs
Scenario 1
throughput {'min': 0.03, 'max': 110.41, 'mean': 57.818561309523815, 'std': 29.608858485445268}
Scenario 2
throughput {'min': 0.22, 'max': 110.35, 'mean': 47.39601785714286, 'std': 26.874653320893362}
Scenario 3
throughput {'min': 0.1, 'max': 109.33, 'mean': 47.72844356261023, 'std': 25.974846578574827}
"""

"""
Scenario 1
throughput {'min': 0.03, 'max': 110.41, 'mean': 57.818561309523815, 'std': 29.608858485445268}
interference {'min': -151.73, 'max': -0.0, 'mean': -97.49411523611523, 'std': 21.936604553912254}
rssi {'min': -81.41, 'max': -37.42, 'mean': -58.52763214285714, 'std': 7.11873413950484}
sinr {'min': -9.4, 'max': 53.64, 'mean': 32.949347619047614, 'std': 8.618997584824925}



Scenario 2
throughput {'min': 0.0, 'max': 57.88, 'mean': 16.012168194980696, 'std': 11.065256101784948}
interference {'min': -150.16, 'max': -0.0, 'mean': -88.64759573476049, 'std': 26.154485706129986}
rssi {'min': -81.41, 'max': -0.0, 'mean': -58.22506696428571, 'std': 6.869791482037659}
sinr {'min': -20.52, 'max': 55.3, 'mean': 30.648423423423424, 'std': 11.37196171194207}



Scenario 3
throughput {'min': 0.0, 'max': 59.59, 'mean': 16.07963606654783, 'std': 11.011395347524639}
interference {'min': -152.24, 'max': -0.0, 'mean': -88.66472291466923, 'std': 26.151377927825887}
rssi {'min': -81.43, 'max': -36.43, 'mean': -58.24301247771835, 'std': 6.8552820047072025}
sinr {'min': -28.39, 'max': 56.5, 'mean': 30.92333821407351, 'std': 10.94405564797456}
"""