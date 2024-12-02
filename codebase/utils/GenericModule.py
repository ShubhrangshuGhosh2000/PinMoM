import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import copy
import numpy as np
import torch
from joblib import dump, load
from utils.ProteinFeaturesHolder import ProteinFeaturesHolder


class GenericModule(object):
    def __init__(self, hyperParams = None):
        if hyperParams is None:
            hyperParams = {}
        self.hyperParams = copy.deepcopy(hyperParams)
        
        if 'featScaleClass' not in self.hyperParams:
            self.featScaleClass = None
        else:
            self.featScaleClass = self.hyperParams['featScaleClass']
        
        self.batchIdx = 0 #necessary for feature selection for some models
        if 'seed' in hyperParams:
            self.seed = int(hyperParams['seed'])
        else:
            self.seed = 1
        self.featuresData = None
        self.model=None
        self.scaleData = None
        self.featDict=  self.hyperParams.get('featLst',{})


    def saveModelToFile(self,fname):
        if self.model is None:
            print('Error, no model to save')
            exit(42)
        else:
            self.model.saveModelToFile(fname)
        self.saveFeatScaler(fname)


    def loadModelFromFile(self,fname):
        if self.model is None:
            self.genModel()
        self.model.loadModel(fname)


    def saveFeatScaler(self,fname):
        if self.scaleData is not None:
            dump(self.scaleData,fname+'_scaler')


    def loadFeatScaler(self,fname):
        for item in self.featDict:
            if os.path.exists(fname+'_scaler'):
                self.scaleData = load(fname+'_scaler')


    def setModel(self,model):
        self.model=model


    def genModel(self):
        pass


    def getModel(self):
        return self.model


    #load from each listed file to each key in the featDict
    def loadFeatureData(self,featureFolder):
        if featureFolder[-1] not in ['/','\\']:
            featureFolder+='/'
        self.featuresData = {}
        for item, fnames in self.featDict.items():
            lst = []
            for fname in fnames:
                lst.append(featureFolder+fname)
            self.featuresData[item] = ProteinFeaturesHolder(lst)


    #by default, load all data into a single 2D matrix, and return it with the class Data
    #if returnDict = True, returns dictionary instead
    def genFeatureData(self,pairs,dataType='train',returnDict=False):
        classData = pairs[:,2]
        classData = classData.astype(np.int)
        lst = []
        d = {}
        for item,dataset in self.featuresData.items():
            lst.append(dataset.genData(pairs))
            d[item] = lst[-1]
        if returnDict:
            return d, classData
        if len(lst) == 1:
            return lst[0], classData
        else:
            return np.hstack(lst), classData


    #used to create validation data from a training set
    def splitFeatures(self,trainFeatures,trainClasses,split=0.1):
    
        #create validation dataset
        trueData = np.where(trainClasses==1)[0]
        falseData = np.where(trainClasses==0)[0]
        np.random.shuffle(trueData)
        np.random.shuffle(falseData)
        validIdx = np.hstack((trueData[0:int(trueData.shape[0]*split)],falseData[0:int(falseData.shape[0]*split)]))
        trainIdx = np.hstack((trueData[int(trueData.shape[0]*split):],falseData[int(falseData.shape[0]*split):]))
        trainFeaturesNew = trainFeatures[trainIdx]
        trainClassesNew = trainClasses[trainIdx]
        validFeaturesNew = trainFeatures[validIdx]
        validClassesNew = trainClasses[validIdx]
        
        return trainFeaturesNew, trainClassesNew, validFeaturesNew, validClassesNew


    def loadEncodingFileWithPadding(self,fileName,maxProteinLength=1200,zeroPadding='right',returnLookup = False):
        zeroPadding = zeroPadding.lower()
        lookupMatrix = []
        f = open(fileName)
        for line in f:
            line = line.strip()
            if len(line)==0:
                break #end of lookup matrix
            lookupMatrix.append([float(k) for k in line.split(',')])
        
        lookupMatrix = torch.tensor(lookupMatrix).long()
        
        #lookup for protein name to row index mapping
        proteinNameMapping = {}
        #list of aaIdx (list of tensors)
        aaIdxs = []
        
        #grab all protein data, and map it to our matrix
        for line in f:
            line = line.strip().split()
            name = line[0]
            aaIdx = torch.tensor([int(k) for k in line[1].split(',')]).long()
            aaIdx = aaIdx[:maxProteinLength]
            if name not in proteinNameMapping:
                proteinNameMapping[name] = len(proteinNameMapping)
                aaIdxs.append(aaIdx)
            else:
                aaIdxs[proteinNameMapping[name]] = aaIdx
        
        #create a torch matrix, will be a 3D array of number of proteins, maxProteinLength, inSize
        dataMatrix = torch.zeros((len(proteinNameMapping),lookupMatrix.shape[1],maxProteinLength))
        for i in range(0,len(aaIdxs)):
            
            #3 dimension indexing:
            #i,  -- protein index
            #: -- lookupMatrix.shape[1]
            #:x.shape[1],  -- protein length being assigned
            
            #each lookup row will be length lookupMatrix.shape[1], creating an N,lookupMatrix.shape[1] shaped matrix
            #transpose to get N to the first dimension
            x = lookupMatrix[aaIdxs[i],:].T
            if zeroPadding == 'right':
                dataMatrix[i,:,:x.shape[1]] = x
            elif zeroPadding == 'left':
                #calculate gap in front of string
                a = dataMatrix.shape[2]-x.shape[1]
                dataMatrix[i,:,a:] = x
            elif zeroPadding == 'edges':
                #calculate total gap
                a = dataMatrix.shape[2]-x.shape[1]
                #start at half gap
                b = a//2
                #end at half gap + sequence length
                c = b + x.shape[1]
                dataMatrix[i,:,b:c] = x
        
        # print('loaded ',dataMatrix.shape[0],'proteins')
        if not returnLookup:
            return proteinNameMapping, dataMatrix
        else:
            return proteinNameMapping, dataMatrix, lookupMatrix


    def loadEncodingFileWithPadding_pd(self,fixed_prot_id,man_2d_feat_dict_prot,maxProteinLength=1200,zeroPadding='right',returnLookup = False):
        zeroPadding = zeroPadding.lower()
        lookupMatrix = man_2d_feat_dict_prot[fixed_prot_id]['SkipGramAA7']['lookupMatrix']
        proteinNameMapping = {}
        aaIdxs = []
        for prot_id, indiv_prot_man_2d_feat_dict in man_2d_feat_dict_prot.items():
            name = prot_id
            aaIdx = indiv_prot_man_2d_feat_dict['SkipGramAA7']['aaIdxs'][0]
            aaIdx = aaIdx[:maxProteinLength]
            if name not in proteinNameMapping:
                proteinNameMapping[name] = len(proteinNameMapping)
                aaIdxs.append(aaIdx)
            else:
                aaIdxs[proteinNameMapping[name]] = aaIdx
        dataMatrix = torch.zeros((len(proteinNameMapping),lookupMatrix.shape[1],maxProteinLength))
        for i in range(0,len(aaIdxs)):
            x = lookupMatrix[aaIdxs[i],:].T
            if zeroPadding == 'right':
                dataMatrix[i,:,:x.shape[1]] = x
            elif zeroPadding == 'left':
                a = dataMatrix.shape[2]-x.shape[1]
                dataMatrix[i,:,a:] = x
            elif zeroPadding == 'edges':
                a = dataMatrix.shape[2]-x.shape[1]
                b = a//2
                c = b + x.shape[1]
                dataMatrix[i,:,b:c] = x
        if not returnLookup:
            return proteinNameMapping, dataMatrix
        else:
            return proteinNameMapping, dataMatrix, lookupMatrix

    
    def loadLabelEncodingFileWithPadding(self,fileName,maxProteinLength=1200,zeroPadding='right',returnLookup = False):
        zeroPadding = zeroPadding.lower()
        lookupMatrix = []  
        f = open(fileName)
        for line in f:
            line = line.strip()
            if len(line)==0:
                break 
            lookupMatrix.append([float(k) for k in line.split(',')])
        lookupMatrix = torch.tensor(lookupMatrix).long()  
        proteinNameMapping = {}
        aaIdxs = []
        for line in f:
            line = line.strip().split()
            name = line[0]
            aaIdx = torch.tensor([int(k)+1 for k in line[1].split(',')]).long()  
            aaIdx = aaIdx[:maxProteinLength]
            if name not in proteinNameMapping:
                proteinNameMapping[name] = len(proteinNameMapping)
                aaIdxs.append(aaIdx)
            else:
                aaIdxs[proteinNameMapping[name]] = aaIdx
        dataMatrix = torch.zeros((len(proteinNameMapping),1,maxProteinLength))
        for i in range(0,len(aaIdxs)):
            x = aaIdxs[i].T
            x = x.reshape(1, len(x))  
            if zeroPadding == 'right':
                dataMatrix[i,:,:x.shape[1]] = x
            elif zeroPadding == 'left':
                a = dataMatrix.shape[2]-x.shape[1]
                dataMatrix[i,:,a:] = x
            elif zeroPadding == 'edges':
                a = dataMatrix.shape[2]-x.shape[1]
                b = a//2
                c = b + x.shape[1]
                dataMatrix[i,:,b:c] = x
        if not returnLookup:
            return proteinNameMapping, dataMatrix
        else:
            return proteinNameMapping, dataMatrix, lookupMatrix

    
    def loadLabelEncodingFileWithPadding_pd(self,fixed_prot_id,man_2d_feat_dict_prot,maxProteinLength=1200,zeroPadding='right',returnLookup = False):
        zeroPadding = zeroPadding.lower()
        lookupMatrix = []  
        lookupMatrix = man_2d_feat_dict_prot[fixed_prot_id]['LabelEncoding']['lookupMatrix']
        proteinNameMapping = {}
        aaIdxs = []
        for prot_id, indiv_prot_man_2d_feat_dict in man_2d_feat_dict_prot.items():
            name = prot_id
            aaIdx = indiv_prot_man_2d_feat_dict['LabelEncoding']['aaIdxs'][0]
            aaIdx = aaIdx[:maxProteinLength]
            if name not in proteinNameMapping:
                proteinNameMapping[name] = len(proteinNameMapping)
                aaIdxs.append(aaIdx)
            else:
                aaIdxs[proteinNameMapping[name]] = aaIdx
        dataMatrix = torch.zeros((len(proteinNameMapping),1,maxProteinLength))
        for i in range(0,len(aaIdxs)):
            x = aaIdxs[i].T
            x = x.reshape(1, len(x))  
            if zeroPadding == 'right':
                dataMatrix[i,:,:x.shape[1]] = x
            elif zeroPadding == 'left':
                a = dataMatrix.shape[2]-x.shape[1]
                dataMatrix[i,:,a:] = x
            elif zeroPadding == 'edges':
                a = dataMatrix.shape[2]-x.shape[1]
                b = a//2
                c = b + x.shape[1]
                dataMatrix[i,:,b:c] = x
        if not returnLookup:
            return proteinNameMapping, dataMatrix
        else:
            return proteinNameMapping, dataMatrix, lookupMatrix


    def scaleFeatures(self,features,scaleType):
        if type(features) is dict:
            return features 
        if self.featScaleClass is not None:
            if scaleType == 'train':
                self.scaleData = self.featScaleClass()
                newFeatures = self.scaleData.fit_transform(features)
                return newFeatures
            else:
                newFeatures = self.scaleData.transform(features)
                return newFeatures
        else:#no scaler, do nothing
            return features


    def setScaleFeatures(self,trainPairs):
        trainFeatures, trainClasses = self.genFeatureData(trainPairs,'train')
        trainFeatures = self.scaleFeatures(trainFeatures,'train')
    

    def train(self,trainPairs):
        trainFeatures, trainClasses = self.genFeatureData(trainPairs,'train')
        trainFeatures = self.scaleFeatures(trainFeatures,'train')
        self.fit(trainFeatures,trainClasses)


    def fit(self,trainFeatures,trainClasses):
        self.genModel() #create a new model from scratch, ensuring we don't overwrite the previously trained one
        self.model.fit(trainFeatures,trainClasses)


    def predictPairs(self, testPairs):
        testFeatures, testClasses = self.genFeatureData(testPairs,'predict')
        testFeatures = self.scaleFeatures(testFeatures,'test')
        return self.predict_proba(testFeatures,testClasses)
    

    def predictPairs_pd(self, testPairs):
        testFeatures = self.genFeatureData_pd(testPairs,'predict')
        return self.predict_proba_pd(testFeatures)


    def predictPairs_xai_DS(self, testPairs, resultsFolderName):
        testFeatures, testClasses = self.genFeatureData(testPairs,'predict')
        testFeatures = self.scaleFeatures(testFeatures,'test')
        return self.predict_proba_xai_DS(testFeatures,testClasses,resultsFolderName)


    def predictPairs_xai_humanBenchmark(self, testPairs, predictFileName):
        testFeatures, testClasses = self.genFeatureData(testPairs,'predict')
        testFeatures = self.scaleFeatures(testFeatures,'test')
        return self.predict_proba_xai_humanBenchmark(testFeatures,testClasses,predictFileName)


    #Predict using batches method.   Assumes all data has been loaded, but computing features for all pairs in memory at once would be infeasible.
    def predictFromBatch(self,testPairs,batchSize):
        predictions = []
        predictClasses = []
        for i in range(0,len(testPairs),batchSize):
            p,c = self.predictPairs(testPairs[i:(i+batchSize),:])
            predictions.append(p)
            predictClasses.append(c)
        return (np.vstack(predictions),np.hstack(predictClasses))
        

    def predictFromFile(self,testFile,batchSize,sep='\t',headerLines=1,classIdx=-1):
        predictions = []
        predictClasses = []
        for header,dataBatch,classBatch in self.parseTxtGenerator(testFile,batchSize,sep,headerLines,classIdx):
            p,c = self.predict_proba(dataBatch,classBatch)
            predictions.append(p)
            predictClasses.append(c)
        return (np.vstack(predictions),np.hstack(predictClasses))
        
        
    def predict_proba(self,predictFeatures,predictClasses):
        preds = self.model.predict_proba(predictFeatures)
        return (preds,np.asarray(predictClasses,dtype=np.int))


    def parseTxtGenerator(self,tsvFile,batch,sep='\t',headerLines=1,classIdx = -1):
        f = open(tsvFile)
        header = None
        curData = []
        classData = []
        for line in f:
            if headerLines >0:
                if header is None:
                    header = line.strip().split(sep)
                else:
                    header = [header]
                    header.append(line.strip().split(sep))
                headerLines -=1
                continue
            line = line.strip().split(sep)
            if classIdx == -1:
                classIdx = len(line)-1
            classData.append(int(line[classIdx]))
            line = line[:classIdx] + line[(classIdx+1):]
            line = [float(s) for s in line]
            curData.append(line)
            if len(curData) == batch:
                yield (header,curData,classData)
                curData =[]
                classData = []
        if len(curData) > 1:
            yield (header,curData,classData)
            curData = []
            classData = []
