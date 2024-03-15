
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model.CausalNetworks import CREncoder, MRDecoderS

def initModel(configs): 
    encoder = getEncoder(configs)
    decoder = getDecoder(configs)
    optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=configs["lr"],
    )
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=configs["lr_decay"], gamma=configs["gamma"]
    )

    return encoder, decoder, optimizer, scheduler



def getEncoder(configs):  
    encoder = CREncoder(
        configs = configs,
        inputDim = configs['Length'] * 1,
        hiddenDim = configs['encoderHidden'],
        outputDim = configs['edgeType'],
        dropOut = configs['encoderDO']
    )
    return encoder.to(configs['device'])


def getDecoder(configs):  
    decoder = MRDecoderS(
        configs = configs,
        decoderHidDim = configs['decoderHidden'],
        dropOut = configs['decoderDO'],
        decConDim = configs['decConDim']
    )    

    return decoder.to(configs['device'])

