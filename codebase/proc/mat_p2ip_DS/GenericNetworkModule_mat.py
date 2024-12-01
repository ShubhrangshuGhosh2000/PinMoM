import sys
from pathlib import Path
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
from utils.GenericModule import GenericModule
import numpy as np

class GenericNetworkModule(GenericModule):
    def __init__(self, hyperParams = {}):
        GenericModule.__init__(self,hyperParams)
        self.dataLookup = None
        self.dataMatrix = None
        self.oneDdataMatrix = None
        self.validationRate = hyperParams.get('ValidationRate',None)
        self.scaleData=None

 
    def fit(self,trainFeatures,trainClasses):
        if self.validationRate is not None and self.validationRate > 0:
            newTrainFeat, newTrainClass, newValidFeat, newValidClass = self.splitFeatures(trainFeatures,trainClasses,self.validationRate)
            self.genModel() 
            self.model.fit(newTrainFeat,newTrainClass,self.dataMatrix, newValidFeat, newValidClass)
        else:
            self.genModel() 
            self.model.fit(trainFeatures,trainClasses,self.dataMatrix,self.oneDdataMatrix)

    
    def predict_proba(self,predictFeatures,predictClasses):
        preds = self.model.predict_proba(predictFeatures,self.dataMatrix, self.oneDdataMatrix)
        return (preds,predictClasses)

  
    def predict_proba_xai_DS(self,predictFeatures,predictClasses,resultsFolderName):
        preds = self.model.predict_proba_xai_DS(predictFeatures,self.dataMatrix, self.oneDdataMatrix, predictClasses, resultsFolderName)
        return (preds,predictClasses)
    
    
    def predict_proba_pd(self,predictFeatures):
        probs = self.model.predict_proba_pd(predictFeatures,self.dataMatrix, self.oneDdataMatrix)
        return probs


    def loadFeatureData(self,featureFolder):
        pass

    
    def scaleFeatures(self,features,scaleType):
        print('no scaling in this model')
        return features


    def saveFeatScaler(self,fname):
        pass


    def loadFeatScaler(self,fname):
        pass

    
    def genFeatureData(self,pairs,dataType='train'):
        classData = np.asarray(pairs[:,2],dtype=np.float32)
        classData = classData.astype(int)
        orgFeatsData = pairs[:,0:2]
        featsData = [self.dataLookup[str(a)] for a in orgFeatsData.flatten()]
        featsData = np.asarray(featsData).reshape(classData.shape[0],2)
        return featsData, classData

    
    def genFeatureData_pd(self,pairs,dataType='train'):
        orgFeatsData = pairs[:,0:2]
        featsData = [self.dataLookup[str(a)] for a in orgFeatsData.flatten()]
        featsData = np.asarray(featsData).reshape(pairs.shape[0],2)
        return featsData


    def predictFromBatch(self,testPairs,batchSize,model=None):
        return self.predictPairs(testPairs,model)

    
    def predictFromFile(self,testFile,batchSize,sep='\t',headerLines=1,classIdx=-1,model=None):
        pass
