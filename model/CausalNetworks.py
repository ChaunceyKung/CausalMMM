import torch
import torch.nn as nn

from model.BaseModules import *



class Encoder(nn.Module):
    def __init__(
        self, 
        configs
    ):
        super(Encoder, self).__init__()
        

    def initParameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                module.bias.data.fill_(0.1)
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



    def edg2Ver(
        self,
        x,
        receiveRelations, 
        sendRelations
    ):
        verInfo = torch.matmul(receiveRelations.t(), x)
        verInfo = verInfo / verInfo.size(1)
        return verInfo
    

    def ver2Edg(
        self,
        x,
        receiveRelations, 
        sendRelations
    ):
        receivInfo = torch.matmul(receiveRelations, x)
        sendInfo = torch.matmul(sendRelations, x)
        edgInfo = torch.cat([receivInfo, sendInfo], dim=2)
        return edgInfo

    def forward(self, inputData, receiveRelations, senRelations):
        return 
    

class CREncoder(Encoder):
    def __init__(self, 
                 configs,
                 inputDim,
                 hiddenDim,
                 outputDim,
                 dropOut=0.0
        ):
        super().__init__(configs)

        self.map1 = BaseMLP(inputDim, hiddenDim, hiddenDim, dropOut)
        self.map2 = BaseMLP(hiddenDim*2, hiddenDim, hiddenDim, dropOut)
        self.map3 = BaseMLP(hiddenDim, hiddenDim, hiddenDim, dropOut)

        self.map4 = BaseMLP(hiddenDim*3, hiddenDim, hiddenDim, dropOut)
        self.mapOut = nn.Linear(hiddenDim, outputDim)
        self.initParameters()
        
    def forward(self, inputData, receiveRelations, sendRelations):

        X = self.map1(inputData)
        X = self.ver2Edg(X, receiveRelations, sendRelations)
        X = self.map2(X)
        
        ResX = X
        X = self.edg2Ver(X, receiveRelations, sendRelations)        
        X = self.map3(X)
        X = self.ver2Edg(X, receiveRelations, sendRelations)
        X = torch.cat((X, ResX), dim = 2  )
        X = self.map4(X)

        return self.mapOut(X)


class MRDecoderS(nn.Module):
    def __init__(self,
                configs,
                decoderHidDim,
                dropOut = 0.0,
                nodeDim = 1, 
                edgeTypes = 2,  
                preSkip = True,
                decConDim = 2
        ):
        super(MRDecoderS, self).__init__()

        self.MSG1 = nn.ModuleList(
            [nn.Linear(2*decoderHidDim, decoderHidDim) for _ in range(edgeTypes)]
        ) 
        self.MSG2 = nn.ModuleList(
            [nn.Linear(decoderHidDim, decoderHidDim) for _ in range(edgeTypes)]
        )

        self.MSGoDim = decoderHidDim
        self.preSkip = preSkip

        self.hidRMap = nn.Linear(decoderHidDim, decoderHidDim, bias=False)
        self.hidIMap = nn.Linear(decoderHidDim, decoderHidDim, bias=False)
        self.hidNMap = nn.Linear(decoderHidDim, decoderHidDim, bias=False)

        self.inRMap = nn.Linear(nodeDim, decoderHidDim, bias=False)
        self.inIMap = nn.Linear(nodeDim, decoderHidDim, bias=False)
        self.inNMap = nn.Linear(nodeDim, decoderHidDim, bias=False)
        
        self.oMap1 = nn.Linear(decoderHidDim, decoderHidDim)
        self.oMap2 = nn.Linear(decoderHidDim, decoderHidDim)
        self.oMap3 = nn.Linear(decoderHidDim, nodeDim)
        self.alphaMap = nn.Linear(decConDim, 1)
        self.gammaMap = nn.Linear(decConDim, 1)
        self.dropOut = dropOut

    def vanillaForward(self, 
                    inputData, 
                    inputCon,
                    receiveRelations, 
                    sendRelations, 
                    edges,  
                    hidden
        ):  
        

        MSGRec = torch.matmul(receiveRelations, hidden)
        MSGSend = torch.matmul(sendRelations, hidden)
        MSGPre = torch.cat([MSGSend, MSGRec], dim=-1)
        MSGContainer = torch.zeros(MSGPre.size(0), MSGPre.size(1), self.MSGoDim )   

        a = self.alphaMap(inputCon).squeeze()
        g = self.gammaMap(inputCon).squeeze()

        aPha = torch.clamp(a, min = 0.1)
        gPha = torch.clamp(g, min = 0.1)

        if inputData.is_cuda:
            MSGContainer = MSGContainer.cuda()
        
        if self.preSkip:
            startIdx = 1
            norm = float(len(self.MSG2)) - 1.0
        else:
            startIdx = 0
            norm = float(len(self.MSG2))
        
        for i in range(startIdx, len(self.MSG2)):
            MSG = torch.tanh(self.MSG1[i](MSGPre))
            MSG = F.dropout(MSG, p=self.dropOut)
            MSG = torch.tanh(self.MSG2[i](MSG))
            MSG = MSG * edges[:, :, i:i+1]
            MSGContainer += MSG / norm
        
        MSGAgg = MSGContainer.transpose(-2, -1).matmul(receiveRelations).transpose(-2, -1)
        MSGAgg = MSGAgg.contiguous() / inputData.size(1)
        inputData = torch.unsqueeze(inputData, -1)
        
        
        hiddenR = torch.sigmoid(self.inRMap(inputData) + self.hidRMap(MSGAgg) ) 
        hiddenI = torch.sigmoid(self.inIMap(inputData) + self.hidIMap(MSGAgg) )
        hiddenN = torch.sigmoid(self.inNMap(inputData) + hiddenR * self.hidNMap(MSGAgg) )
        
        hidden = (1 - hiddenI) * hiddenN + hiddenI * hidden
        
        pred = F.dropout(F.relu(self.oMap1(hidden)), p=self.dropOut)
        pred = F.dropout(F.relu(self.oMap2(pred)), p=self.dropOut)
        pred = self.oMap3(pred)   
        
        force = torch.clamp(pred[:,-1, :], min = 0.1)
        satPred = ( torch.pow(force, aPha   )     )  / ( torch.pow(force, aPha) + torch.pow(gPha, aPha) ) 
        
        satPred *=  pred[:,-1, :]
        satPred = satPred.unsqueeze(0)
   
        predRes = torch.cat([pred[:,:-1,:], satPred], dim=1)
 
        return predRes, hidden


    def forward(
        self,
        data,
        cont,
        edges,
        receiveRelations, 
        sendRelations, 
        MLength, 
        TMLength 
    ):
      
        inputData = data.transpose(1,2).contiguous()
        TLength = inputData.size(1)
        hidden = torch.zeros(inputData.size(0), inputData.size(2), self.MSGoDim)
        if inputData.is_cuda:
            hidden = hidden.cuda()
        
        predContainer = []
        
        for step in range(0, inputData.size(1) - 1):
            if step <= TMLength:
                inputDataVanilla = inputData[:, step, :]
            else:
                inputDataVanilla = predContainer[step - 1]
        
            pred, hidden = self.vanillaForward(
                inputDataVanilla, 
                cont,
                receiveRelations, 
                sendRelations, 
                edges,  
                hidden
            )
            predContainer.append(pred[:, :, 0])
        
        predRes = torch.stack(predContainer, dim=1)
        predRes = predRes.transpose(1, 2).contiguous()
        
        return predRes

