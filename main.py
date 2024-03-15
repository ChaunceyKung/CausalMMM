
import time
import numpy as np
import torch
from collections import defaultdict
from model.forward_pass import forwardPassEval

from model import modelInit


from utils import *
from loader import *

def train(
    configs, 
    trainLoader, 
    validLoader,
    encoder, 
    decoder,  
    optimizer,  
    scheduler,  
    globalLogPrior, 
    logger 
):

    bestValLoss = np.inf
    bestEpoch = 0
    
    receiveRelations, sendRelations = buildReceiveSendRelations(configs)
    
    for epoch in range(configs['epochs']):
        epochStartTime = time.time()
        trainLosses = defaultdict(list)

        for batchIdx, miniBatch in enumerate(trainLoader):
            data, relations, context = unpackMiniBatches(configs, miniBatch)  
            optimizer.zero_grad()
            
            lossDict, decoderOutput, edgeSamples = forwardPassEval(
                configs,
                encoder, 
                decoder,
                data,
                context,
                relations,
                receiveRelations,
                sendRelations,
                globalLogPrior
            )

            loss = lossDict["trainingLoss"]
            loss.backward()
            optimizer.step()

            trainLosses = appendLosses(trainLosses, lossDict)

        logString = logger.resultString("train", epoch, trainLosses, t=epochStartTime)
        logger.write2LogFile(logString)
        logger.appendTrainLoss(trainLosses)
        scheduler.step()
            
        if configs["validate"]:
            valLosses = val(
                epoch,
                configs, 
                validLoader,
                encoder, 
                decoder,  
                globalLogPrior, 
                logger, 
                receiveRelations,
                sendRelations
            )
            valLoss = np.mean(valLosses["trainingLoss"]  ) 
            if valLoss < bestValLoss:
                print("Best model so far on validation set, saving...")
                logger.createLog(
                    encoder,
                    decoder,
                    np.mean(trainLosses["AUROC"]),
                    optimizer,
                    False,
                    None
                )
                bestValLoss = valLoss
                bestEpoch = epoch
        elif (epoch + 1) % 50 == 0: 
            logger.createLog(
                encoder,
                decoder,
                np.mean(trainLosses["AUROC"]),
                optimizer,
                False,
                None
            )                    
        
        print("Training steps:")
        print("ACC Loss is") 
        print(np.mean(trainLosses["ACC"]))
        print(np.std(trainLosses["ACC"]))
        print("AUROC Loss is") 
        print(np.mean(trainLosses["AUROC"]))
        print(np.std(trainLosses["AUROC"]))
   
    return bestEpoch, epoch
          
        

def val(
    epoch,
    configs, 
    validLoader,
    encoder,  
    decoder,  
    globalLogPrior, 
    logger, 
    receiveRelations,
    sendRelations
):
    valStartTime = time.time()
    valLosses = defaultdict(list)

    encoder.eval()
    decoder.eval()

    for batchIdx, miniBatch in enumerate(validLoader): 
        data, relations, context = unpackMiniBatches(configs, miniBatch)
        with torch.no_grad():
            lossDict, decoderOutput, edgeSamples = forwardPassEval(
                configs,
                encoder, 
                decoder,
                data,
                context,
                relations,
                receiveRelations,
                sendRelations,
                globalLogPrior
            )
        valLosses = appendLosses(valLosses, lossDict)
    
    logString = logger.resultString("validate", epoch, valLosses, t=valStartTime)
    logger.write2LogFile(logString)
    logger.appendValLoss(valLosses)

    encoder.train()
    decoder.train()

    return valLosses


if __name__ == "__main__":
    args = initArg()
    configs = loadConfig(args.config_file)
    setup_seed(100)
    fullDataSet = synDataFromNpy(configs["dataPath"], configs["relationsPath"], configs["contextPath"])
    trainSize = int(len(fullDataSet) * 0.8)
    validSize = len(fullDataSet) - trainSize  

    trainDataSet, validDataSet = torch.utils.data.random_split(fullDataSet, [trainSize, validSize])

    trainLoader = DataLoader.DataLoader(trainDataSet, batch_size=configs["BS"], shuffle=False) 
    validLoader = DataLoader.DataLoader(validDataSet, batch_size=configs["BS"], shuffle=False) 

    encoder, decoder, optimizer, scheduler = modelInit.initModel(configs)

    globalLogPrior = None
    logs = Logger(configs)    
    
    bestEpoch, epoch = train(
        configs, 
        trainLoader, 
        validLoader,
        encoder,  
        decoder, 
        optimizer,  
        scheduler,  
        globalLogPrior, 
        logs 
    )

    print("Optimization Finished!")

