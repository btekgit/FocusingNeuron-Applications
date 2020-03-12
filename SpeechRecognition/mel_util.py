import librosa
import numpy as np                
import pandas as pd


from functools import partial
sr = 16000
iLen = 16000
inputLength = 16000


mspct = partial(librosa.feature.melspectrogram,sr=sr, S=None, n_fft=1024, hop_length=128, center=True, pad_mode='reflect', power=1.0, n_mels=80,
                             fmin=40.0, fmax=sr/2)
#x_train_mfcc = np.zeros(shape=(x_train.shape[0],125,80,1), dtype=np.float16)
#for i,x in enumerate(x_train[0:10]):



def shapeFix(curX,dim=16000):
    fX = np.zeros(shape=dim, dtype=curX.dtype)
    if curX.shape[0] == dim:
        fX = curX
        #print('Same dim')
    elif curX.shape[0] > dim: #bigger
        #we can choose any position in curX-self.dim
        randPos = np.random.randint(curX.shape[0]-dim)
        fX = curX[randPos:randPos+dim]
        #print('File dim bigger')
    else: #smaller
        randPos = np.random.randint(dim-curX.shape[0])
        fX[randPos:randPos+curX.shape[0]] = curX
        
    return fX


# read and convert all files.

def extractMELSandDump(path):
    allfiles  = []
    alldirs  = []
    for root, dirs, files in os.walk(path):
        allfiles += [root+'/'+ f for f in files if f.endswith('.wav.npy')]
        alldirs += [root+'/' for f in files if f.endswith('.wav.npy')]

    N=125

    for fn,dn in zip(allfiles,alldirs):
        x = np.load(fn) # load the file
        x = shapeFix(x)
        #print(x.shape)

        # EXTRACT MFCC FEATURES
        x_mfcc=mspct(y=x)
        x_mfcc_crop = x_mfcc[:,:,np.newaxis] # crop and add axis
        x_mfcc_crop = np.transpose(librosa.power_to_db(x_mfcc_crop),(1, 0, 2))
        x_base = np.zeros((N,x_mfcc_crop.shape[1],1))
        #print(x_mfcc_crop.shape)
        if(x_mfcc_crop.shape[0]<N):
            x_base[N-x_mfcc_crop.shape[0]:,:,0]= x_mfcc_crop[:,:,:]
        elif(x_mfcc_crop.shape[0]>N):
            x_base = x_mfcc_crop[x_mfcc_crop.shape[0]-N:,:,:]
        else:
            x_base = x_mfcc_crop
            #x_mfcc_crop = np.pad(x_mfcc_crop,(125-x_mfcc_crop.shape[0])//2,'edge')
        print(x_base.shape)
        assert(x_base.shape[0]==125)

        # preparet the output filename
        wavname = fn.split('/')[-1]
        mfccname =wavname.split('.')[:-2][0]+'.mfcc.npy'
        #print(fn.split('/')[:-1],)
        print(dn+mfccname)
        np.save(dn+mfccname,x_base,allow_pickle=True)
