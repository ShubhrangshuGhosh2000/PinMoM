import os,sys
import pandas as pd
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_DS_test import MatP2ipNetworkModule
import numpy as np
from utils import PPIPUtils
import time


def writePredictions(fname,predictions,classes):
    f = open(fname,'w')
    for i in range(0,predictions.shape[0]):
        f.write(str(predictions[i])+'\t'+str(classes[i])+'\n')
    f.close()

def writeScore(predictions,classes, fOut, predictionsFName=None, thresholds=[0.01,0.03,0.05,0.1,0.25,0.5,1]):
    finalPredictions = np.hstack(predictions)
    finalClasses = np.hstack(classes)
    results = PPIPUtils.calcScores(finalClasses,finalPredictions,thresholds)
    lst = PPIPUtils.formatScores(results,'Total')
    for line in lst:
        fOut.write('\t'.join(str(s) for s in line) + '\n')
    fOut.write('\n')
    if predictionsFName is not None:
        writePredictions(predictionsFName,finalPredictions,finalClasses)


def runTest_matpip(model, outResultsName,trainSets,testSets,featureFolder,hyperParams = {},predictionsName =None,loadedModel= None,modelsLst = None
                   ,resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None, startIdx=0,loads=None
                   ,fixed_prot_id=None,feature_data=None):
    model.loadFeatureData_matpip(fixed_prot_id, feature_data)
    preds = model.predictPairs_pd(testSets[0])
    class_1_prob_arr = 1.0 - preds[:,1]
    return class_1_prob_arr


def runTrainOnly_DS(modelClass, trainSets, featureFolder, hyperParams, saveModels, spec_type):
    t = time.time()
    if featureFolder[-1] != '/':
        featureFolder += '/'
    model = modelClass(hyperParams)
    model.loadFeatureData_DS(featureFolder, spec_type)
    model.train(trainSets[0])
    if saveModels is not None:
        model.saveModelToFile(saveModels[0])
    return model


def runTest_DS(modelClass, outResultsName,trainSets,testSets,featureFolder,hyperParams = {},predictionsName =None,loadedModel= None,modelsLst = None,resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None, startIdx=0,loads=None,spec_type=None):
    t = time.time()
    if featureFolder[-1] != '/':
        featureFolder += '/'
    outResultDf = None
    outResultCsvFileName = outResultsName.replace('.txt', '.csv')
    if resultsAppend and outResultsName:
        outResults = open(outResultsName,'a')
    elif outResultsName:
        outResults = open(outResultsName,'w')
    totalPredictions = []
    totalClasses = []
    trainedModelsLst = []
    for i in range(0,startIdx):
        totalPredictions.append([])
        totalClasses.append([])
        trainedModelsLst.append([])
    if loadedModel is not None:
        model = loadedModel
    else:
        model = modelClass(hyperParams)
        model.loadFeatureData_DS(featureFolder, spec_type)
    for i in range(startIdx,len(testSets)):
        model.batchIdx = i
        if modelsLst is None:
            model.batchIdx = i
            if loads is None or loads[i] is None:
                model.train(trainSets[i])
                if saveModels is not None:
                    model.saveModelToFile(saveModels[i])
            else:
                model.loadModelFromFile(loads[i])
            preds, classes = model.predictPairs(testSets[i])
            if keepModels:
                trainedModelsLst.append(model.getModel())
        else:
            preds, classes = model.predictPairs(testSets[i],modelsLst[i])
            if keepModels:
                trainedModelsLst.append(modelsLst[i])
        results = PPIPUtils.calcScores_DS(classes,preds[:,1])
        lst = PPIPUtils.formatScores_DS(results,'Species: '+spec_type)
        if outResultsName:
            for line in lst:
                outResults.write('\t'.join(str(s) for s in line) + '\n')
            outResults.write('\n')
            outResults.close()
            outResults = open(outResultsName,'a')
            score_df = pd.DataFrame({'Species': [spec_type]
                            , 'AUPR': [results['AUPR']], 'Precision': [results['Precision']]
                            , 'Recall': [results['Recall']], 'AUROC': [results['AUROC']]
                            })
            outResultDf = score_df if outResultDf is None else pd.concat([outResultDf, score_df], axis=0, sort=False)
            outResultDf.to_csv(outResultCsvFileName, index=False)
        
        totalPredictions.append(preds[:,1])
        totalClasses.append(classes)
        if predictionsFLst is not None:
            writePredictions(predictionsFLst[i],totalPredictions[i],totalClasses[i])
    if outResultsName:
        outResults.write('Time: '+str(time.time()-t))
        outResults.close()
    return (totalPredictions, totalClasses, model, trainedModelsLst)


def runTest_DS_xai(modelClass,testSets,featureFolder,hyperParams = {},startIdx=0,loads=None,spec_type=None,resultsFolderName=None):
    if featureFolder[-1] != '/':
        featureFolder += '/'
    model = modelClass(hyperParams)
    model.loadFeatureData_DS(featureFolder, spec_type)
    for i in range(startIdx,len(testSets)):
        model.loadModelFromFile(loads[i])
        model.predictPairs_xai_DS(testSets[i],resultsFolderName)

