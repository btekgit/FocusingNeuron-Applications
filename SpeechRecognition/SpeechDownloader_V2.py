"""
File containing scripts to download audio from various datasets

Also has tools to convert audio into numpy

updates by BTEK 14 Jan 2020
1) SpeechDownloader_v2.py now runs with basePath


2) SpeechDownloader_v2.py now uses stratified split. Labels equally distributed. 

3) Did some cleaning SpeechDownloader_v2.py . Removed wav2numpy pass. I guess one time is ok.

updates by BTEK 19 JAN 2020

4) Reads any extension not only .wav.npy, e.g. mfcc.npy
"""
from tqdm import tqdm
import requests
import math
import os
import tarfile
import numpy as np
import librosa
import pandas as pd

import audioUtils
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


###################
# Google Speech Commands Dataset V2
###################

#GSCmdV2Categs = {'unknown' : 0, 'silence' : 1, '_unknown_' : 0, '_silence_' : 1, '_background_noise_' : 1, 'yes' : 2, 
#                 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11}
#numGSCmdV2Categs = 12

#"Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", 
#"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine"

GSCmdV2Categs = {'unknown' : 0, 'silence' : 0, '_unknown_' : 0, '_silence_' : 0, '_background_noise_' : 0, 'yes' : 2, 
                 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11,
                 'zero' : 12, 'one' : 13, 'two' : 14, 'three' : 15, 'four' : 16, 'five' : 17, 'six' : 18, 
                 'seven' : 19,  'eight' : 20, 'nine' : 1 }
numGSCmdV2Categs = 21

def PrepareGoogleSpeechCmd(version = 2, forceDownload = False, 
                           task = '20cmd', nTestSplit=0.1, 
                           nValSplit=0.1, basePath=None, rseed=42,
                           ext='.wav.npy',verbose=False):
    
    """
    Prepares Google Speech commands dataset version 2 for use
    
    tasks: 20cmd, 12cmd, leftright or 35word
    
    Returns full path to training, validation and test file list and file categories
    """
    allowedTasks = ['12cmd', 'leftright', '35word', '20cmd']
    if task not in allowedTasks:
        raise Exception('Task must be one of: {}'.format(allowedTasks))

###########################################################################################
####BTEK added this to show different base Path 
##########################################################################################         
    if not basePath:
        if version == 2:
            _DownloadGoogleSpeechCmdV2(forceDownload)
            basePath = 'sd_GSCmdV2'
        elif version == 1:
            _DownloadGoogleSpeechCmdV1(forceDownload)
            basePath = 'sd_GSCmdV1'
        else:
            raise Exception('Version must be 1 or 2')
        
    if task == '12cmd':
        GSCmdV2Categs = {'unknown' : 0, 'silence' : 1, '_unknown_' : 0, '_silence_' : 1, '_background_noise_' : 1, 'yes' : 2, 
                 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11}
        numGSCmdV2Categs = 12
    elif task=='leftright':
        GSCmdV2Categs = {'unknown' : 0, 'silence' : 0, '_unknown_' : 0, '_silence_' : 0, '_background_noise_' : 0, 
                         'left' : 1, 'right' : 2}
        numGSCmdV2Categs = 3
    elif task=='35word':
        GSCmdV2Categs = {'unknown' : 0, 'silence' : 0, '_unknown_' : 0, '_silence_' : 0, '_background_noise_' : 0, 'yes' : 2, 
                         'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11,
                         'zero' : 12, 'one' : 13, 'two' : 14, 'three' : 15, 'four' : 16, 'five' : 17, 'six' : 18, 
                         'seven' : 19,  'eight' : 20, 'nine' : 1, 'backward':21, 'bed':22, 'bird':23, 'cat':24, 'dog':25,
                         'follow':26, 'forward':27, 'happy':28, 'house':29, 'learn':30, 'marvin':31, 'sheila':32, 'tree':33,
                         'visual':34, 'wow':35}
        numGSCmdV2Categs = 36
    elif task=='20cmd':
        GSCmdV2Categs = {'unknown' : 0, 'silence' : 0, '_unknown_' : 0, '_silence_' : 0, '_background_noise_' : 0, 'yes' : 2, 
                         'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11,
                         'zero' : 12, 'one' : 13, 'two' : 14, 'three' : 15, 'four' : 16, 'five' : 17, 'six' : 18, 
                         'seven' : 19,  'eight' : 20, 'nine' : 1 }
        numGSCmdV2Categs = 21
        
###########################################################################################
####BTEK commented this. As we do not redo it at each run.
########################################################################################## 
    if verbose:
        print('By passing converting test set WAVs to numpy files. Run once!')
        print('By passing Converting training set WAVs to numpy files')
#    audioUtils.WAV2Numpy(basePath + '/test/')
#    audioUtils.WAV2Numpy(basePath + '/train/')
##########################################################################################
####YT  changed the test, train and validation data creation to random distribution instead of fixed data
##########################################################################################
    #read split from files and all files in folders
 #   testWAVs = pd.read_csv(basePath+'/train/testing_list.txt', sep=" ", header=None)[0].tolist()
 #   valWAVs  = pd.read_csv(basePath+'/train/validation_list.txt', sep=" ", header=None)[0].tolist()

 #   testWAVs = [os.path.join(basePath+'/train/', f + '.npy') for f in testWAVs if f.endswith('.wav')]
 #   valWAVs  = [os.path.join(basePath+'/train/', f + '.npy') for f in valWAVs if f.endswith('.wav')]
    allWAVs  = []
    for root, dirs, files in os.walk(basePath+'/train/'):
        allWAVs += [root+'/'+ f for f in files if f.endswith(ext)]
    
    ###########################################################################################
    ####BTEK added strafied split
    ##########################################################################################
    
    tmplabels    = [_getFileCategory(f, GSCmdV2Categs) for f in allWAVs]
    if verbose:
        print("All train instances: ",len(tmplabels), ", existing labels:", np.unique(tmplabels))

    trainAllWAVs, testWAVs = train_test_split(allWAVs, shuffle=True, 
                                              test_size=nTestSplit, 
                                              random_state=rseed, stratify=tmplabels)
    
    tmplabels    = [_getFileCategory(f, GSCmdV2Categs) for f in trainAllWAVs]
    trainWAVs, valWAVs = train_test_split(trainAllWAVs, shuffle=True, 
                                          test_size=nValSplit, 
                                          random_state=rseed*2, stratify=tmplabels)
    
    #Check if val and train are overllapping. it should not be. 
    assert(not valWAVs[0] in trainWAVs)
    assert(not testWAVs[0] in trainWAVs)
#    trainWAVs = list( set(allWAVs)-set(valWAVs)-set(testWAVs) )

    testWAVsREAL = []
    for root, dirs, files in os.walk(basePath+'/test/'):
        testWAVsREAL += [root+'/'+ f for f in files if f.endswith(ext)]

    #get categories
    testWAVlabels     = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVs]
    valWAVlabels      = [_getFileCategory(f, GSCmdV2Categs) for f in valWAVs]
    trainWAVlabels    = [_getFileCategory(f, GSCmdV2Categs) for f in trainWAVs]
    testWAVREALlabels = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVsREAL]
    
    #background noise should be used for validation as well
    backNoiseFiles = [trainWAVs[i] for i in range(len(trainWAVlabels)) if trainWAVlabels[i]==GSCmdV2Categs['silence']]
    backNoiseCats  = [GSCmdV2Categs['silence'] for i in range(len(backNoiseFiles))]
    if numGSCmdV2Categs==12:
        valWAVs += backNoiseFiles
        valWAVlabels += backNoiseCats

    
    #build dictionaries
    testWAVlabelsDict     = dict(zip(testWAVs, testWAVlabels))
    valWAVlabelsDict      = dict(zip(valWAVs, valWAVlabels))
    trainWAVlabelsDict    = dict(zip(trainWAVs, trainWAVlabels))
    testWAVREALlabelsDict = dict(zip(testWAVsREAL, testWAVREALlabels))
    
    #a tweak here: we will heavily underuse silence samples because there are few files.
    #we can add them to the training list to reuse them multiple times
    #note that since we already added the files to the label dicts we don't need to do it again
    
    #for i in range(200):
    #    trainWAVs = trainWAVs + backNoiseFiles
    
    #info dictionary
    trainInfo = {'files' : trainWAVs, 'labels' : trainWAVlabelsDict, 'n' : len(trainWAVs), 'dist': np.histogram(trainWAVlabels,np.unique(trainWAVlabels))}
    valInfo = {'files' : valWAVs, 'labels' : valWAVlabelsDict, 'n' : len(valWAVs), 'dist': np.histogram(valWAVlabels,np.unique(valWAVlabels))}
    testInfo = {'files' : testWAVs, 'labels' : testWAVlabelsDict, 'n' : len(testWAVs), 'dist': np.histogram(testWAVlabels,np.unique(testWAVlabels))}
    testREALInfo = {'files' : testWAVsREAL, 'labels' : testWAVREALlabelsDict, 'n' : len(testWAVsREAL), 'dist':np.histogram(testWAVREALlabels,np.unique(testWAVREALlabels))}
    gscInfo = {'train' : trainInfo, 'test' : testInfo, 'val' : valInfo, 'testREAL' : testREALInfo} 
    if verbose:
        print("train set: ", trainInfo['n'], " labels dist:", trainInfo['dist'][0])
        print("val set: ", valInfo['n']," labels dist:", valInfo['dist'])
        print("test set: ", testInfo['n']," labels dist:", testInfo['dist'])
        print("test set real: ", testREALInfo['n']," labels dist:", testREALInfo['dist'])
		
        print('Done preparing Google Speech commands dataset version {}'.format(version))
    
    return gscInfo, numGSCmdV2Categs
    
    
def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_GSCmdV2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ,0)


def _DownloadGoogleSpeechCmdV2(forceDownload = False):
    """
    Downloads Google Speech commands dataset version 2
    """
    if os.path.isdir("sd_GSCmdV2/") and not forceDownload:
        print('Google Speech commands dataset version 2 already exists. Skipping download.')
    else:
        if not os.path.exists("sd_GSCmdV2/"):
            os.makedirs("sd_GSCmdV2/")
        trainFiles = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        testFiles = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
        _downloadFile(testFiles, 'sd_GSCmdV2/test.tar.gz')
        _downloadFile(trainFiles, 'sd_GSCmdV2/train.tar.gz')
    
    #extract files
    if not os.path.isdir("sd_GSCmdV2/test/"):
        _extractTar('sd_GSCmdV2/test.tar.gz', 'sd_GSCmdV2/test/')
        
    if not os.path.isdir("sd_GSCmdV2/train/"):
        _extractTar('sd_GSCmdV2/train.tar.gz', 'sd_GSCmdV2/train/')
        
        
def _DownloadGoogleSpeechCmdV1(forceDownload = False):
    """
    Downloads Google Speech commands dataset version 1
    """
    if os.path.isdir("sd_GSCmdV1/") and not forceDownload:
        print('Google Speech commands dataset version 1 already exists. Skipping download.')
    else:
        if not os.path.exists("sd_GSCmdV1/"):
            os.makedirs("sd_GSCmdV1/")
        trainFiles = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
        testFiles = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz'
        _downloadFile(testFiles, 'sd_GSCmdV1/test.tar.gz')
        _downloadFile(trainFiles, 'sd_GSCmdV1/train.tar.gz')
    
    #extract files
    if not os.path.isdir("sd_GSCmdV1/test/"):
        _extractTar('sd_GSCmdV1/test.tar.gz', 'sd_GSCmdV1/test/')
        
    if not os.path.isdir("sd_GSCmdV1/train/"):
        _extractTar('sd_GSCmdV1/train.tar.gz', 'sd_GSCmdV1/train/')        

##############
# Utilities
##############

def _downloadFile(url, fName):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0)); 
    block_size = 1024
    wrote = 0 
    print('Downloading {} into {}'.format(url, fName))
    with open(fName, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        
def _extractTar(fname, folder):
    print('Extracting {} into {}'.format(fname, folder))
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=folder)
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path=folder)
        tar.close()      
