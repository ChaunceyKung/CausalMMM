import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils import *

def forwardPassEval(
    configs,
    encoder,
    decoder,
    data,
    contexts,
    relations,        
    receiveRelations,  
    sendRelations,    
    globalLogPrior 
):

    startTime = time.time()
    lossDict = defaultdict(lambda: torch.zeros((), device=configs['deviceType']))

    edgeHidden = encoder(data, receiveRelations, sendRelations) 
    

    edgeSamples = gumbelSoftmaxSample(edgeHidden, tau = configs['tau'])
    edgeProb = F.softmax(edgeHidden, dim=-1)  

    targetData = data[:, :, 1:]

    decoderOutput = decoder(
        data = data,
        cont = contexts,
        edges = edgeSamples,
        receiveRelations = receiveRelations, 
        sendRelations = sendRelations, 
        MLength = configs['predSteps'], 
        TMLength = configs['Length'] - configs['predSteps'] # T-M  
    )

    lossDict["ACC"] = edgeWiseACC(edgeHidden, relations)

    lossDict["AUROC"] = edgeWiseAUROC(edgeProb, relations)
    lossDict["mseLoss"] = F.mse_loss(decoderOutput, targetData)
    lossDict["rmseLoss"] = torch.sqrt(lossDict["mseLoss"])
    criterion = nn.L1Loss()
    lossDict["maeLoss"] = criterion(decoderOutput, targetData)
    
    lossDict["klLoss"] = klLoss(configs, edgeProb, globalLogPrior)
    lossDict["nllLoss"] = nllGaussianLoss(decoderOutput, targetData, configs["nllVar"])
    totalLoss = configs['lambda'] * lossDict["klLoss"] + lossDict["nllLoss"]
    lossDict["trainingLoss"] = totalLoss

    lossDict["inferenceTime"] = time.time() - startTime

    return lossDict, decoderOutput, edgeSamples


        

