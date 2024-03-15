# configs loader & data loader

import datetime
import torch
import argparse
import numpy as np
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

def initArg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str)
    return parser.parse_args()


def loadConfig(cfg_file = "./configs/configs.txt"):
    # cfg_file = "./configs/configs.txt"
    f = open(cfg_file, 'r')
    config_lines = f.readlines()
    cfgs = { }
    for line in config_lines:
        ps = [p.strip() for p in line.split('=')]
        if (len(ps) != 2):
            continue
        try:
            if (ps[1].find(',') != -1):
                str_line = ps[1].split(',')
                cfgs[ps[0]] = list(map(int, str_line))
            elif (ps[1].find('.') == -1):
                cfgs[ps[0]] = int(ps[1])
            else:
                cfgs[ps[0]] = float(ps[1])
        except ValueError:
            cfgs[ps[0]] = ps[1]
            if cfgs[ps[0]] == 'False':
                cfgs[ps[0]] = False
            elif cfgs[ps[0]] == 'True':
                cfgs[ps[0]] = True         

    cfgs["time"] = datetime.datetime.now().isoformat()
    cfgs["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfgs["cuda"] = torch.cuda.is_available()


    return cfgs


class synDataFromNpy(Dataset.Dataset):
    def __init__(
        self, 
        dataPath,
        relationsPath,
        contextPath
    ):
        self.data = np.load(dataPath)
        self.relations = np.load(relationsPath)
        self.context = np.load(contextPath)
    def __getitem__(self, item):
        return torch.from_numpy(self.data[item]).to(torch.float32), torch.from_numpy(self.relations[item]).to(torch.float32), torch.from_numpy(self.context[item]).to(torch.float32)
    def __len__(self):
        return self.data.shape[0]

