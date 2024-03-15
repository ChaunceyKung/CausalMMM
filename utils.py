import numpy as np
import os
import time
import math
import torch
import random
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import json

import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def unpackMiniBatches(args, minibatch):
    (data, relations, context) = minibatch
    if args['cuda']:
        data, relations, context = data.cuda(), relations.cuda(), context.cuda()        
    return data, relations, context



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


def sample4GumbelDistribution(shape, eps=1e-10):
    Z = torch.rand(shape).float()
    gumbelDistribution = -torch.log(eps - torch.log(Z + eps))
    return gumbelDistribution


def gumbelSoftmaxSample(edgesHidden, tau=0.5, eps=1e-10):
    gumbelDistribution = sample4GumbelDistribution(edgesHidden.size(), eps=eps)
    if edgesHidden.is_cuda:
        gumbelDistribution = gumbelDistribution.cuda()
    res = edgesHidden + Variable(gumbelDistribution)
    res = F.softmax(res / tau, dim=-1)
    return res




def klLoss(configs, edgeProb, globalLogPrior):
    klDivergence = 0.0
    varNum = configs['varNum']
    eps=1e-16

    if configs['usePrior'] != 1: 
        klDivergence = edgeProb * (torch.log(edgeProb + eps) - globalLogPrior)
    else: 
        klDivergence = edgeProb * (torch.log(edgeProb + eps))
    klRes = klDivergence.sum() / (varNum * edgeProb.size(0))
    return klRes


def edgeWiseACC(edgePreds, edgeTruths, binary=True):  
    _, edgePreds = edgePreds.max(-1)
    if binary:
        edgePreds = (edgePreds >= 1).long()
    rightStat = edgePreds.float().data.eq(edgeTruths.float().data.view_as(edgePreds)).cpu().sum()
    return np.float(rightStat) / (edgeTruths.size(0) * edgeTruths.size(1))

def edgeWiseAUROC(edgePreds, edgeTruths):  
    aurocScore = 0.0
    edgePreds = 1 - edgePreds[:, :, 0]
    aurocScore = roc_auc_score(
        edgeTruths.cpu().detach().flatten(),
        edgePreds.cpu().detach().flatten()
    )
    return aurocScore
    

def nllGaussianLoss(decoderOutput, targetData, variance):
    negLogProb = (decoderOutput - targetData) ** 2 / (2 * variance)
    nllRes = negLogProb.sum() / (targetData.size(0) * targetData.size(1))
    return nllRes

def appendLosses(lossesList, losses):
    for loss, value in losses.items():
        if type(value) == float:
            lossesList[loss].append(value)
        elif type(value) == defaultdict:
            if lossesList[loss] == []:
                lossesList[loss] = defaultdict(list)
            for idx, elem in value.items():
                lossesList[loss][idx].append(elem)
        else:
            lossesList[loss].append(value.item())
    return lossesList


def onehotMapper(categories):
    cSets = set(categories)
    cDict = {c: np.identity(len(cSets))[i, :] for i,c in enumerate(cSets)}
    onehotMapRes = np.array(list(map(cDict.get, categories)), dtype=np.int32)
    return onehotMapRes
    

def buildReceiveSendRelations(configs): 
    varNum = configs["varNum"]
    offDiagMatrix = np.ones([varNum, varNum]) - np.eye(varNum)
    receiveRelations = np.array(onehotMapper(np.where(offDiagMatrix)[0]), dtype=np.float32 )
    receiveRelations = torch.FloatTensor(receiveRelations)
    sendRelations = np.array(onehotMapper(np.where(offDiagMatrix)[1]), dtype=np.float32 )
    sendRelations = torch.FloatTensor(sendRelations)

    if configs["cuda"]:
        receiveRelations = receiveRelations.cuda()
        sendRelations = sendRelations.cuda()

    return receiveRelations, sendRelations    

class Logger:
    def __init__(self, configs):
        self.configs = configs

        self.train_losses = pd.DataFrame()
        self.train_losses_idx = 0

        self.test_losses = pd.DataFrame()
        self.test_losses_idx = 0

        if configs["validate"]:
            self.val_losses = pd.DataFrame()
            self.val_losses_idx = 0
        else:
            self.val_losses = None

        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

        self.createLogPath(configs)

    def createLogPath(self, configs, add_path_var=""):

        log_path = os.path.join(configs["saveFolder"], add_path_var, configs["time"])
        self.log_path = log_path

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        self.log_file = os.path.join(log_path, "log.txt")

        self.encoder_file = os.path.join(log_path, "encoder.pt")
        self.decoder_file = os.path.join(log_path, "decoder.pt")
        self.optimizer_file = os.path.join(log_path, "optimizer.pt")

        self.plotdir = os.path.join(log_path, "plots")
        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)

    def saveCheckpoint(self, encoder, decoder, optimizer, specifier=""):
        self.encoder_file = os.path.join(self.log_path, "encoder" + specifier + ".pt")
        self.decoder_file = os.path.join(self.log_path, "decoder" + specifier + ".pt")
        self.optimizer_file = os.path.join(
            self.log_path, "optimizer" + specifier + ".pt"
        )

        if encoder is not None:
            torch.save(encoder.state_dict(), self.encoder_file)
        if decoder is not None:
            torch.save(decoder.state_dict(), self.decoder_file)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), self.optimizer_file)

    def write2LogFile(self, string):

        print(string)
        cur_file = open(self.log_file, "a")
        print(string, file=cur_file)
        cur_file.close()

    def createLog(
        self,
        encoder=None,
        decoder=None,
        auroc=None,
        optimizer=None,
        final_test=False,
        test_losses=None,
    ):

        print("Saving model and log-file to " + self.log_path)

        self.train_losses.to_pickle(os.path.join(self.log_path, "train_loss"))

        if self.val_losses is not None:
            self.val_losses.to_pickle(os.path.join(self.log_path, "val_loss"))

        if auroc is not None:
            np.save(os.path.join(self.log_path, "auroc"), auroc)

        specifier = ""
        if final_test:
            pd_test_losses = pd.DataFrame(
                [
                    [k] + [np.mean(v)]
                    for k, v in test_losses.items()
                    if type(v) != defaultdict
                ],
                columns=["loss", "score"],
            )
            pd_test_losses.to_pickle(os.path.join(self.log_path, "test_loss"))

            pd_test_losses_per_influenced = pd.DataFrame(
                list(
                    itertools.chain(
                        *[
                            [
                                [k]
                                + [idx]
                                + [np.mean(list(itertools.chain.from_iterable(elem)))]
                                for idx, elem in sorted(v.items())
                            ]
                            for k, v in test_losses.items()
                            if type(v) == defaultdict
                        ]
                    )
                ),
                columns=["loss", "num_influenced", "score"],
            )
            pd_test_losses_per_influenced.to_pickle(
                os.path.join(self.args.log_path, "test_loss_per_influenced")
            )

            specifier = "final"

        # Save the model checkpoint
        self.saveCheckpoint(encoder, decoder, optimizer, specifier=specifier)

    def drawLossCurves(self):
        for i in self.train_losses.columns:
            plt.figure()
            plt.plot(self.train_losses[i], "-b", label="train " + i)

            if self.val_losses is not None and i in self.val_losses:
                plt.plot(self.val_losses[i], "-r", label="val " + i)

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc="upper right")

            # save image
            plt.savefig(os.path.join(self.args.log_path, i + ".png"))
            plt.close()

    def appendTrainLoss(self, loss):
        for k, v in loss.items():
            self.train_losses.at[str(self.train_losses_idx), k] = np.mean(v)
        self.train_losses_idx += 1

    def appendValLoss(self, val_loss):
        for k, v in val_loss.items():
            self.val_losses.at[str(self.val_losses_idx), k] = np.mean(v)
        self.val_losses_idx += 1

    def appendTestLoss(self, test_loss):
        for k, v in test_loss.items():
            if type(v) != defaultdict:
                self.test_losses.at[str(self.test_losses_idx), k] = np.mean(v)
        self.test_losses_idx += 1

    def resultString(self, trainvaltest, epoch, losses, t=None):
        string = ""
        if trainvaltest == "test":
            string += (
                "-------------------------------- \n"
                "--------Testing----------------- \n"
                "-------------------------------- \n"
            )
        else:
            string += str(epoch) + " " + trainvaltest + "\t \t"

        for loss, value in losses.items():
            if type(value) == defaultdict:
                string += loss + " "
                for idx, elem in sorted(value.items()):
                    string += str(idx) + ": {:.10f} \t".format(
                        np.mean(list(itertools.chain.from_iterable(elem)))
                    )
            elif np.mean(value) != 0 and not math.isnan(np.mean(value)):
                string += loss + " {:.10f} \t".format(np.mean(value))

        if t is not None:
            string += "time: {:.4f}s \t".format(time.time() - t)

        return string



