import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import torch
from utils.AACounter import AACounter

#1 epoch is more than enough to train a network this small with enough proteins
def LabelEncoding(fastas, fileName, labelEncode_lookup_prsnt= False, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'],groupLen=1,sorting=False, flip=False,excludeSame=False,deviceType='cpu'):
    if groupings is not None:
        groupMap = {}
        idx = 0
        for item in groupings:
            for let in item:
                groupMap[let] = idx
            idx += 1
        numgroups = len(groupings)
        # print(groupMap,numgroups)
    else:
        groupMap = None
        numgroups=20
    parsedData = AACounter(fastas, groupMap, groupLen, sorting=sorting,flip=flip,excludeSame=excludeSame,getRawValues=True,deviceType=deviceType)
    lookupMatrix = []
    if(not labelEncode_lookup_prsnt):
        #number of unique groups, typically 7, times length of our embeddings, typically 1, equals the corpus size
        corpusSize = numgroups*groupLen
        for i in range(0,corpusSize):
            lst = [0] * corpusSize
            lst[i] = 1
            lookupMatrix.append(lst)
        lookupMatrix = torch.tensor(lookupMatrix).long()
    proteinNameMapping = {}
    aaIdxs = []
    #grab all protein data, and map it to our matrix
    for item in parsedData[1:]:
        name = item[0]
        stVals = item[1].cpu().numpy()
        aaIdx = torch.tensor([int(k)+1 for k in stVals]).long()
        if name not in proteinNameMapping:
            proteinNameMapping[name] = len(proteinNameMapping)
            aaIdxs.append(aaIdx)
        else:
            aaIdxs[proteinNameMapping[name]] = aaIdx
    labelEnc_feat_dict = {'lookupMatrix': lookupMatrix, 'proteinNameMapping': proteinNameMapping, 'aaIdxs': aaIdxs}
    return labelEnc_feat_dict
