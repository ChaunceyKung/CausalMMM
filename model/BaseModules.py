
import torch.nn as nn
import torch.nn.functional as F
import math



class BaseMLP(nn.Module):
    def __init__(
        self,
        inputDim,
        hiddenDim,
        outputDim,
        dropOut=0.0,
        batchNorm=True        
    ):
        super(BaseMLP, self).__init__()
        self.linearLayer1 = nn.Linear(inputDim, hiddenDim)
        self.linearLayer2 = nn.Linear(hiddenDim, outputDim)
        self.batchNormLayer = nn.BatchNorm1d(outputDim)
        self.batchNorm = batchNorm
        self.dropOut = dropOut
        self.initParameters()

    def initParameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                module.bias.data.fill_(0.1)
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def batchNormFunc(self, inputData):
        x = inputData.view(inputData.size(0) * inputData.size(1), -1)
        x = self.batchNormLayer(x)
        return x.view(inputData.size(0), inputData.size(1), -1)

    def forward(self, inputData):
        x = F.elu(self.linearLayer1(inputData))
        x = F.dropout(x, self.dropOut, training=self.training)
        x = F.elu(self.linearLayer2(x))
        if self.batchNorm:
            return self.batchNormFunc(x)
        else:
            return x
