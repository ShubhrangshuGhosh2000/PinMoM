import torch
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from utils.PreprocessUtils import getAALookup


def calcI(lst,numGroups):
    x = lst[0]
    for i in range(1,len(lst)):
        x = x * numGroups + lst[i]
    return x

#Amino Acid Composition, set to take N groups (default is each of the 20 regular amino acid in its own group) and a group len (default of 1)
#will count the number of times each unique group of length group length occurs
#if sorting is True, if using group length of 3, then sequences AGV, AVG, VAG, GAV, GVA, and VGA will all group to the same bin
#if flip is True and sorting is False, then sequences AGV and VGA will match, sequences VAG and GAV will match, and sequences GVA and AVG will match, but all six will not group to the same bin
#if exclude same, when group length % 2 == 0, if the grouping found is the same group repeated (such as 2,2 or  3,6,3,6), it will not be counted.  
#if getRawValues is true, return a list containing 1 tensor per string, where the tensor values map to the group's idx.  Also returns group mapping
def AACounter(fastas, groupings = None, groupLen=1, sorting=False,flip=False,normType='100',separate=False,excludeSame=False,getRawValues=False,flattenRawValues=True,gap=0,truncate='right',masking = None, maskWeights = None, deviceType='cpu'):
    deviceType = 'cpu'
    AA =getAALookup()
    if groupings is not None:
        AA = groupings
    allGroups = set()
    for k,v in AA.items():
        allGroups.add(v)
    numGroups = len(allGroups)
    retData = []
    if separate:
        rawData = []
    if groupLen%2 != 0 : 
        excludeSame= False
    maskingMatrix = None
    if masking: 
        if maskWeights is not None and len(maskWeights) != len(masking):
            print('Error, mask weights must be the same length as masking')
            exit(42)
        maskLen = len(masking[0])
        for item in masking:
            if len(item) != maskLen:
                print('Error, all masks must be the same length')
                exit(42)
            if item.count('1') != groupLen:
                print('Error, non-group length mask found')
                exit(42)
        maskingMatrix = []
        for i in range(0,maskLen):
            maskingMatrix.append([])
        for i in range(0,len(masking)):
            for j in range(0,maskLen):
                if masking[i][j] == '1':
                    maskingMatrix[j].append(i)
        for i in range(0,maskLen):
            maskingMatrix[i] = torch.tensor(maskingMatrix[i]).to(deviceType)
    else:
        maskWeights = None
    if gap > 0:
        truncate = truncate.lower()
        if truncate not in ['right','left','midright','midleft']:
            print('error, unrecognized gap type')
            exit(42)
    header = ['protein']
    comboNum = numGroups**groupLen
    groupMapping = {}
    groupLookup = []
    for i in range(0,comboNum):
        idxs = []
        for j in range(0,groupLen):
            idxs.append((i//(numGroups**j))%numGroups)
        idxs = idxs[::-1]
        if sorting:
            if sorted(idxs) != idxs:
                groupLookup.append(groupMapping[calcI(sorted(idxs),numGroups)])
                continue
        elif flip:
            if idxs > idxs[::-1]:
                groupLookup.append(groupMapping[calcI(idxs[::-1],numGroups)])
                continue
        if excludeSame:
            if idxs[len(idxs)//2:] == idxs[:len(idxs)//2]:
                groupLookup.append(-1)
                continue
        header.append('g_'+'_'.join(str(k) for k in idxs))
        groupLookup.append(len(groupMapping))
        groupMapping[calcI(idxs,numGroups)] = len(groupMapping)
    if separate:
        retData.append([header[0]])
        rawData.append(header[1:])
    else:
        retData.append(header)
    groupLookup = torch.tensor(groupLookup)
    groupLookup[groupLookup==-1] = len(groupMapping)
    idx = -1
    for item in fastas:
        idx += 1
        name = item[0]
        st = item[1]
        stIdx = torch.tensor([AA.get(st[k],-1) for k in range(0,len(st))]).to(deviceType).long()
        stIdx = stIdx[stIdx!=-1]
        fastaVals = [name]
        if masking is not None:
            tData = torch.zeros(stIdx.shape[0]-(maskLen-1),device=deviceType).long()
            tDataMat = torch.zeros((len(masking),tData.shape[0])).long()
            for i in range(0,maskLen):
                tDataMat[maskingMatrix[i]] = tDataMat[maskingMatrix[i]] * numGroups + stIdx[i:(stIdx.shape[0]-(maskLen-i-1))]
            tData = tDataMat.T 
        else:
            tData = torch.zeros(stIdx.shape[0]-(groupLen-1),device=deviceType).long()
            for i in range(0,groupLen):
                tData = tData * numGroups + stIdx[i:(stIdx.shape[0]-(groupLen-i-1))]
        tData2 = groupLookup[tData.flatten()].reshape(tData.shape)
        if gap > 0:
            extra = (tData2.shape[0]-1) - (((tData2.shape[0]-1)//gap)*gap)
            if extra > 0:
                extra1 = 0
                extra2 = 0
                if truncate == 'right':
                    extra2 = extra
                elif truncate == 'left':
                    extra1 = extra
                elif truncate == 'midleft':
                    extra2 == extra//2
                    extra1 = extra-extra2
                elif truncate == 'midright':
                    extra1 = extra//2
                    extra2 = extra-extra1
                tData2 = tData2[extra1:(tData2.shape[0]-extra2)]
            tData2 = tData2[torch.arange(0,tData2.shape[0],gap)]
        if getRawValues: 
            if masking is None or flattenRawValues:
                tData2 = tData2.flatten()
                tData2 = tData2[tData2!=len(groupMapping)]
            fastaVals.append(tData2)
            retData.append(fastaVals)
            continue
        binCounts = torch.zeros(len(groupMapping)+1,device=deviceType)
        if maskWeights:
            for i in range(0,len(maskWeights)):
                binCounts2 = torch.zeros(binCounts.shape,device=deviceType)
                z = tData2[:,i]
                binCounts2.index_add_(0,z,torch.ones(z.shape,device=deviceType))
                binCounts += binCounts2 * maskWeights[i]
        else:
            tData2 = tData2.flatten() 
            binCounts.index_add_(0,tData2,torch.ones(tData2.shape[0],device=deviceType))
        binCounts = binCounts[:-1]
        if normType == '100':
            binCounts = binCounts/binCounts.sum()
        elif normType == 'CTD':            
            m1 = binCounts.min()
            m2 = binCounts.max()
            binCounts = (binCounts-m1)/m2
        elif normType == 'SeqLen': 
            divisor = stIdx.shape[0]-(groupLen-1)
            binCounts = binCounts/divisor
        elif normType is None:
            pass 
        binCounts = binCounts.tolist()
        if not separate:
            fastaVals += binCounts
            retData.append(fastaVals)
        else:
            rawData.append(binCounts)
            retData.append(fastaVals)
    if separate:
        retData = (retData,rawData)
    return retData
