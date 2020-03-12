"""
    VERSION 15
1) Cleaned the commented unused code
2) met_util.py created and the related code moved to that file
3) Bug for the model - Querry output comp is fixed
4) It can now generate the confusion matrix taking the output of the test data. Loads the test data for the matrix
5) Loads Real test data for commp. graphs instead of validation data
"""

import os
if os.name=="nt" :
    basepath = 'C:\\data\\isikun\\KWS'
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    outputpath = 'output\\'
else :
    os.environ['CUDA_VISIBLE_DEVICES']="2"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    basepath = '/media/home/rdata/audio/sd_GSCmdV1'
    outputpath = 'output/'




import tensorflow as tf

from tensorflow.python.client import device_lib


import matplotlib
import numpy as np                
import pandas as pd

import matplotlib.pyplot as plt

import SpeechDownloader_V2
import SpeechGenerator
import SpeechModels
import mel_util



from keras.models import Model, load_model,Sequential
from keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax, TimeDistributed
#from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAverageMaxPooling1D,GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling1D,GlobalMaxPooling1D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import Add, Dropout, BatchNormalization, Conv2D, Reshape, MaxPooling2D, Dense, CuDNNLSTM, Bidirectional
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras import regularizers

from kapre.time_frequency import Melspectrogram, Spectrogram

from kapre.utils import Normalization2D

from Kfocusing import FocusedLayer1D
from keras import layers as L
from keras import initializers
np.random.seed(42)

sr = 16000
iLen = 16000
inputLength = 16000



# In[9]:


####  YT    #########################################
#  Replaces AttentionRNN Function with the FN Model according to the following parameters
####  YT    #########################################

def FNSpeechModel(settings):
    
    #simple LSTM
    K.clear_session()
    sr = samplingrate = 16000
    iLen = 16000
    inputLength = 16000
    input_shape = (125,80,1)
    rnn_func = L.CuDNNGRU
    
    inputs = Input(input_shape, name='input')               #[N,1600]

    x = Conv2D(10, (5,1) , activation='relu', padding='same') (inputs) #[N,125,80,10]
    x = BatchNormalization() (x)                                  #[N,125,80,10]
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)  #[N,125,80,1]
    x = BatchNormalization() (x)                                  #[N,125,80,1]

    #x = Reshape((125, 80)) (x)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis) #[N,125,80]

    x = Bidirectional(rnn_func(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim] [N,125,128]
    x = Bidirectional(rnn_func(64, return_sequences = True)) (x) # [b_s, seq_len, vec_dim] [N,125,128] 
    #x = Dropout(0.25)(x)
    
    if (settings['brep_att_focus']):
        # direct replica of dense
        
        xf = Lambda(lambda q: q[:,64]) (x) #[b_s, vec_dim] [N,128]
        query = FocusedLayer1D(units=settings['nhidden'],
                               name='focus-att',
                               activation='linear',
                               init_sigma=settings['focus_init_sigma'], 
                               init_mu=settings['focus_init_mu'],
                               init_w= None,
                               train_sigma=settings['focus_train_si'], 
                               train_weights=settings['focus_train_weights'],
                               si_regularizer=settings['focus_sigma_reg'],
                               train_mu = settings['focus_train_mu'],
                               normed=settings['focus_norm_type'],
                               use_bias=False)(xf)      #Added in V6
        #query = BatchNormalization() (query)                                  #[N,125,80,1]
        attScores = Dot(axes=[1,2])([query, x]) #[N,128].[N,125,128] = [N,125]
    elif (settings['brep_att_dense']) :
        if (settings['bFlipDim']) :
            xf = Permute((2, 1))(x) #[N,128,125]            
        else :
            xf = x
        
        init_mu = settings['focus_init_mu']
        query = FocusedLayer1D(units=128,
                               name='focus-att',
                               activation='linear',
                               init_sigma=settings['focus_init_sigma'], 
                               init_mu=init_mu,
                               init_w= None,
                               train_sigma=settings['focus_train_si'], 
                               train_weights=settings['focus_train_weights'],
                               si_regularizer=settings['focus_sigma_reg'],
                               train_mu = settings['focus_train_mu'],
                               normed=settings['focus_norm_type'],
                               kernel_regularizer=regularizers.l1(1e-7),
                               use_bias=True,
                               perrow=True)(xf)
        #qf = TimeDistributed(FocusedLayer1D(units=1,
        #                       name='focus-unit',
        #                       activation='linear',
        #                       init_sigma=settings['focus_init_sigma'], 
        #                       init_mu=init_mu,
        #                       init_w= None,
        #                       train_sigma=settings['focus_train_si'], 
        #                       train_weights=settings['focus_train_weights'],
        #                       si_regularizer=settings['focus_sigma_reg'],
        #                       train_mu = settings['focus_train_mu'],
        #                       normed=settings['focus_norm_type'],
        #                       use_bias=True),name="focus-att")(xf)      #Added in V6
        #print("QF shape", qf.shape)
        #query = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_this') (qf)
        #print("Query shape", query.shape)

#   
        
        attScores = Dot(axes=[1,2])([query,x]) #[N,128].[N,125,128] = [N,125]
        print("Attscores shape", attScores.shape)
        #attScores = K.sum(attScores,axis=-1)#K.tf.linalg.diag_part(attScores)
        
    else:
        xFirst = Lambda(lambda q: q[:,64]) (x) #[b_s, vec_dim] [N,128]
        query = Dense(128) (xFirst)   #[N,128]
        attScores = Dot(axes=[1,2])([query, x]) #[N,128].[N,125,128] = [N,125]

    if (settings['brep_att_vector']) :
        attVector = FocusedLayer1D(units=settings['nhidden'],
                               name='focus-att-vector',
                               activation='linear',
                               init_sigma= settings['focus_init_sigma'], 
                               init_mu=settings['focus_init_mu'],
                               init_w= None,
                               train_sigma=settings['focus_train_si'], 
                               train_weights=settings['focus_train_weights'],
                               si_regularizer=settings['focus_sigma_reg'],
                               train_mu = settings['focus_train_mu'],
                               normed=settings['focus_norm_type'])(query)    

    else :
        #dot product attention
        #attScores = Dot(axes=[1,2])([query, x]) #[N,128].[N,125,128] = [N,125]
        attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len] [N,125]

        #rescale sequence
        attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim] [N,125}.[N,125,128] = [N,128]

        
    if (settings['brep_Out_Dense']):
        x = FocusedLayer1D(units=64,
                               name='focus-out1',
                               activation='linear',
                               init_sigma=settings['focus_init_sigma'], 
                               init_mu=settings['focus_init_mu'],
                               init_w= None,
                               train_sigma=settings['focus_train_si'], 
                               train_weights=settings['focus_train_weights'],
                               si_regularizer=settings['focus_sigma_reg'],
                               train_mu = settings['focus_train_mu'],
                               normed=settings['focus_norm_type'])(attVector)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = FocusedLayer1D(units=32,
                               name='focus-out2',
                               activation='linear',
                               init_sigma=settings['focus_init_sigma'], 
                               init_mu=settings['focus_init_mu'],
                               init_w= None,
                               train_sigma=settings['focus_train_si'], 
                               train_weights=settings['focus_train_weights'],
                               si_regularizer=settings['focus_sigma_reg'],
                               train_mu = settings['focus_train_mu'],
                               normed=settings['focus_norm_type'])(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
    else:
        x = Dense(64, activation = 'relu')(attVector)   #[N,64]
        #x = Dropout(0.2)(x)
        x = Dense(32)(x)                                #[N,32]
        #x = Dropout(0.2)(x)

    output = Dense(nCategs, activation = 'softmax', name='output')(x)    #[N,21]

    model = Model(inputs=[inputs], outputs=[output])
    
    return model


# In[10]:


#####################################################################################
#######YT Updated FN to run both models on the same data
######################################################################################
import keras_utils
from keras_utils import AdamwithClip
from keras_utils import SGDwithLR, PrintLayerVariableStats
from datetime import datetime




def RunBoth(data,settings={},rseed = 42) :
#####################################################################################
#######YT parameters : settings['cnn_model']        if True builds the model by updating the ConvSpeechModel
####                                                else builds the model by updating the AttentionRNN Model
####                   settings['Optimizer']        SGD or Adam
######################################################################################
    if (settings['model']=='conv_model'):                            
        model = ConvSpeechModel (settings)
    elif settings['model']=='paper':
        model = SpeechModels.AttRNNSpeechModelMELSPEC(nCategs, samplingrate = sr, inputShape = (125,80,1), rnn_func=L.CuDNNGRU)#, rnn_func=L.LSTM)
        if settings['Optimizer'] =='SGD':
            opt = optimizers.SGD(lr=settings['lr_all'], momentum=0.9, clipvalue=1.0)
        else:
            opt = optimizers.Adam(clipvalue=1.0)
        
    elif settings['model']=='focused':
        model = FNSpeechModel(settings)
    
        if settings['Optimizer'] =='SGD':
            lr_dict = {'all':settings['lr_all'],'Sigma':0.01,'Mu':0.01}
            mom_dict= {'all':0.9}
            clip_dict = {'Sigma': [0.01, 2.0],'Mu':[0.01,0.99]}
            decay_dict = {'all':0.9}
            e_i = data[0].shape[0]/settings['batch_size']
            decay_epochs =np.array([e_i*20,e_i*20], dtype='int64')

            opt = SGDwithLR(lr=lr_dict, momentum = mom_dict, decay=decay_dict,
                        clips=clip_dict,decay_epochs=decay_epochs, 
                        verbose=1,clipvalue=1.0)   #, update_clip=1.0 not used as gives error

        else :
           clip_dict = {'Sigma': [0.05, 2.0], 'Mu':[0.01,0.99]}
           opt = AdamwithClip(clips=clip_dict,clipvalue=1.0)
    
    #print(model.trainable_weights)
    model.compile(optimizer= opt, loss=['sparse_categorical_crossentropy'], 
                  metrics=['accuracy'])      
    model.summary()
    model.save('epoch0_fo_'+str(rseed)+'.h5')
    #print(" Number of batches per epoch: ", trainGen.__len__())
    
    # BTEK: added memory data
    x_train,y_train,x_val,y_val,x_test,y_test,x_testR, y_testR = data
    
    # checkpoint
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    filepath=outputpath+settings['model']+str(rseed)+'best_weights'+timestr+'.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
    
    callbacks = [checkpoint]
    if settings['brep_att_dense']:
        stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
        stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
        if settings['model']=='focused' and settings['verbose']:
            pr_1 = PrintLayerVariableStats("focus-att","Weights:0",stat_func_list,stat_func_name)
            pr_2 = PrintLayerVariableStats("focus-att","Sigma:0",stat_func_list,stat_func_name)
            pr_3 = PrintLayerVariableStats("focus-att","Mu:0",stat_func_list,stat_func_name)
            callbacks+=[pr_1,pr_2,pr_3]
    
    results  = model.fit(x_train,y_train, epochs=settings['Epochs'], 
                                   validation_data=(x_val,y_val), 
                                   batch_size=settings['batch_size'],
                         callbacks=callbacks, verbose=1, shuffle=True)
    model.load_weights(filepath)
    score = model.evaluate(x_test,y_test, verbose=1)
    scoreR = model.evaluate(x_testR, y_testR, verbose=1)
    print("Test:",score,"TestR:",scoreR)
    return score, scoreR, results, model


# ## PARAMETER SETTINGS
# will be updated to read from a file so that multiple runs could be done 

# ## RUNNING THE BATCHES

# In[11]:


def repeated_trials(testFunc, nTest, FName, settings):
    list_scores =[]
    list_real_scores = []
    list_histories =[]
    np.random.seed(42)
    for i in range(nTest):
        print("--------------------------------------------------------------")
        print("REPEAT:",i)
        print("--------------------------------------------------------------")
        
        sc, scr, hs, ms = testFunc(settings,rseed=31+i*17) # testten final epoch score, history ve model dönüyor. 
        list_scores.append(sc)
        list_real_scores.append(scr)
        list_histories.append(hs)
        
    print("Final test scores", list_scores)
    print("Final real data scores", list_real_scores)
    mx_scores = [np.max(list_histories[i].history['val_acc']) for i in range(len(list_histories))]
    histories = [h.history for h in list_histories]
    print("Max sscores", mx_scores)
    print("Max stats", np.mean(mx_scores), np.std(mx_scores))
    
    import matplotlib.pyplot as plt
    val_acc = np.array([ v['val_acc'] for v in histories])
    mn = np.mean(val_acc,axis=0)
    st = np.std(val_acc,axis=0)
    
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
#ks = str(kwargs['kernel_size'])+'x'+str(kwargs['kernel_size'])
    filename = 'output\\'+FName+timestr+'_'+'_results.npz'
    np.savez_compressed(filename,mod=settings, mx_scr = mx_scores, hists=histories, list_scr=list_scores, list_real_scr=list_real_scores)  # BURADA ISTEDİĞİN KEY İLE KAYDEDEBİLİRSİN!

    
    plt.plot(mn,linewidth=2.0)
    plt.fill_between(np.linspace(0,mn.shape[0],mn.shape[0]),y1=mn-st,y2=mn+st, alpha=0.25) 
    return mx_scores, histories, list_scores, list_real_scores, ms


mod={'dset': None, 'model':'focused', 'nhidden':128, 
'nfilters':(32,32), 'kn_size':(5,5),
'focus_init_sigma':0.250,   #0.025,
'focus_init_mu': 0.50,#'spread',#0.5, #np.linspace(0.1, 0.9, 128), #veya 0.5 for center
'focus_train_mu':True, 
'focus_train_si':True,'focus_train_weights':True,'focus_norm_type':0,
'focus_sigma_reg':None,'augment':False, 
'Epochs':20, 'batch_size':64,'repeats':1,
'lr_all':0.02,
'brep_att_dense':True,                #if True replaces the Dense but uses GlobalAveragePool to merge. 
'brep_att_focus':False,                # if True directly replaces the DENSE [:,64]
'bFlipDim' : True,                   #if True changes the dimensions of the voice data (x,y) to (y,x)
'brep_Out_Dense': True,              #if True replaces the Dense in output section
'nOutLayer':1,                             #nuımber of focusing layers in the output section
'brep_att_vector':False,               #if True replaces the whole attention vector operations with FN
 'nhiddenAtt':128,  
'Optimizer' :'SGD',
'verbose':True}                  #Adamwithclip or SGDwithLR from keras_utils


# In[13]:


def getData(gscInfo,nCategs):
    trainGen = SpeechGenerator.SpeechGen(gscInfo['train']['files'], gscInfo['train']['labels'],  
                                     batch_size=gscInfo['train']['n'], shuffle=True, dim=(125,80,1),mels=True)
    valGen   = SpeechGenerator.SpeechGen(gscInfo['val']['files'], gscInfo['val']['labels'], 
                                     batch_size=gscInfo['val']['n'], shuffle=True, dim=(125,80,1),mels=True)
    testGen  = SpeechGenerator.SpeechGen(gscInfo['test']['files'], gscInfo['test']['labels'], shuffle=False,
                                     batch_size=gscInfo['test']['n'], dim=(125,80,1),mels=True)
    testRGen = SpeechGenerator.SpeechGen(gscInfo['testREAL']['files'], gscInfo['testREAL']['labels'], 
                                     shuffle=False, batch_size=gscInfo['testREAL']['n'], dim=(125,80,1),mels=True)

    x_train,y_train = trainGen.__getitem__(0)
    x_val,y_val = valGen.__getitem__(0)
    x_test,y_test = testGen.__getitem__(0)
    x_testR,y_testR = testRGen.__getitem__(0)
    
    # norm it here.
    
    #mn_tr = np.mean(x_train,axis=0)
    #st_tr = np.std(x_train,axis=0)
    
    mn_tr = np.mean(x_train) # use single mean and st to prevent exploding empty values. 
    st_tr = np.std(x_train)
    
    
    def norm_x(x_t,mn,st):
        x_t-=mn
        x_t= x_t/(1e-10+st)
        return x_t
    
    x_train=norm_x(x_train,mn_tr,st_tr)
    x_val=norm_x(x_val,mn_tr,st_tr)
    x_test=norm_x(x_test,mn_tr,st_tr)
    x_testR=norm_x(x_testR,mn_tr,st_tr)
    
    # this is just for unit test! REMOVE IT
    #x_train=x_train[:100]
    #y_train=y_train[:100] 
    return x_train,y_train,x_val,y_val,x_test,y_test,x_testR, y_testR


# In[ ]:


#####################################################################################
#######YT V8 Modified version of repeated_trials function to run both of the models with the same data.
######     NOT a NEAT CODE, need to tidy up later
######
######                     MAIN 
######################################################################################

#basepath = 'C:\\Users\\yasem\\projects\\SpeechCmdRecognition-master\\SpeechCmdRecognition-master\\sd_GSCmdV1'


list_scoresP =[]
list_real_scoresP = []
list_historiesP =[]
list_scoresFN =[]
list_real_scoresFN = []
list_historiesFN =[]
rseed = 42
np.random.seed(rseed)



for i in range(mod['repeats']):
    print("--------------------------------------------------------------")
    print("REPEAT:",i)
    print("--------------------------------------------------------------")
 
    rseed = rseed+1
    gscInfo, nCategs = SpeechDownloader_V2.PrepareGoogleSpeechCmd(basePath=basepath,
                                                                  version=1, 
                                                                  task = '20cmd',
                                                                  nTestSplit=0.1,
                                                                  nValSplit=0.1, rseed=rseed, ext='.mfcc.npy')
    data = getData(gscInfo,nCategs)
    
    #####  YT run for Focused
    mod['model']='focused'
    
    scFN, scrFN, hsFN,  msFN  = RunBoth(data,settings=mod,rseed=31+i*17) # testten final epoch score, history ve model dönüyor. 
    
    list_scoresFN.append(scFN)
    list_real_scoresFN.append(scrFN)
    list_historiesFN.append(hsFN)

    ##### YT V8 run for Paper
    mod['model']='paper'
    
    
    scP, scrP, hsP,  msP  = RunBoth(data,settings=mod,rseed=31+i*17) # testten final epoch score, history ve model dönüyor. 
    list_scoresP.append(scP)
    list_real_scoresP.append(scrP)
    list_historiesP.append(hsP)
        
##### YT V8 print results
        
print("Final test scores FN", list_scoresFN)
print("Final real data scores FN", list_real_scoresFN)
mx_val_scoresFN = [np.max(list_historiesFN[i].history['val_acc']) for i in range(len(list_historiesFN))]
historiesFN = [h.history for h in list_historiesFN]
print("Max val scores", mx_val_scoresFN)
print("Mean+Std stats", np.mean(mx_val_scoresFN), np.std(mx_val_scoresFN))

print("Final test scores Paper", list_scoresP)
print("Final real data scores", list_real_scoresP)
mx_val_scoresP = [np.max(list_historiesP[i].history['val_acc']) for i in range(len(list_historiesP))]
historiesP = [h.history for h in list_historiesP]
print("Max val accuracy", mx_val_scoresP)
print("Mean+Std: val stats", np.mean(mx_val_scoresP), np.std(mx_val_scoresP))


#####################################################################################
#######YT V8 save results to files
######################################################################################
now = datetime.now()
timestr = now.strftime("%Y%m%d-%H%M%S")
#ks = str(kwargs['kernel_size'])+'x'+str(kwargs['kernel_size'])

FName="Focused_"
filename = outputpath+FName+timestr+'_'+'_results.npz'
np.savez_compressed(filename,mod=mod, mx_scr = mx_val_scoresFN, hists=historiesFN, list_scr=list_scoresFN, list_real_scr=list_real_scoresFN)  # BURADA ISTEDİĞİN KEY İLE KAYDEDEBİLİRSİN!

FName="Paper_"
filename = outputpath+FName+timestr+'_'+'_results.npz'
np.savez_compressed(filename,mod=mod, mx_scr = mx_val_scoresP, hists=historiesP, list_scr=list_scoresP, list_real_scr=list_real_scoresP)  # BURADA ISTEDİĞİN KEY İLE KAYDEDEBİLİRSİN!

#####################################################################################
#######YT V8 SaVE models
######################################################################################
# SAVING THE LAST MODEL
msFN.save(outputpath+'fn_model_save_'+timestr+'.h5')  # creates a HDF5 file 'my_model.h5'
msP.save(outputpath+'paper_model_save_'+timestr+'.h5')  # creates a HDF5 file 'my_model.h5'
msFN.save_weights('fn_model_save_model_weights'+timestr+'.h5')
msP.save_weights('paper_model_save_model_weights'+timestr+'.h5')
# In[ ]:
from keras.utils import plot_model
plot_model(msFN, to_file='images/model-FN.png')
plot_model(msFN, to_file='images/model-P.png')


# In[ ]:
#RUN RECORD
print(timestr)
import os
os.system('ipython nbconvert --to html yourNotebook.ipynb')


# In[ ]:


#Plot VAL MEAN+STD    
import matplotlib.pyplot as plt

val_accP = np.array([ v['val_acc'] for v in historiesP])
val_accFn = np.array([ v['val_acc'] for v in historiesFN])

mnP = np.mean(val_accP,axis=0)
stP = np.std(val_accP,axis=0)
mnFn = np.mean(val_accFn,axis=0)
stFn = np.std(val_accFn,axis=0)

plt.plot(mnP,'r-+',linewidth=2.0)
plt.fill_between(np.linspace(0,mnP.shape[0],mnP.shape[0]),y1=mnP-stP,y2=mnP+stP, alpha=0.25) 
plt.plot(mnFn,'g-o',linewidth=2.0)
plt.fill_between(np.linspace(0,mnFn.shape[0],mnFn.shape[0]),y1=mnFn-stFn,y2=mnFn+stFn, alpha=0.25) 
plt.legend(loc='best')
plt.ylabel('mean accuracy')
plt.xlabel('epoch')
plt.title('Mean of validation accuracy')


# In[ ]:


mx_P = np.max (mx_val_scoresP)
mx_PMean = np.mean (mx_val_scoresP)
mx_PSTD = np.std (mx_val_scoresP)

list_scoresPa = np.array(list_scoresP)[:,1]
scP = np.max(list_scoresPa)
scPMean = np.mean(list_scoresPa)
scPSTD = np.std(list_scoresPa)

list_real_scoresPa = np.array(list_real_scoresP)[:,1]
scRP = np.max (list_real_scoresPa)
scRPMean = np.mean(list_real_scoresPa)
scRPSTD = np.std(list_real_scoresPa)

mx_N = np.max (mx_val_scoresFN)
mx_NMean = np.mean (mx_val_scoresFN)
mx_NSTD = np.std (mx_val_scoresFN)

list_scoresFNa = np.array(list_scoresFN)[:,1]
scN = np.max(list_scoresFNa)
scNMean = np.mean(list_scoresFNa)
scNSTD = np.std(list_scoresFNa)

list_real_scoresFNa = np.array(list_real_scoresFN)[:,1]
scRN = np.max (list_real_scoresFNa)
scRNMean = np.mean(list_real_scoresFNa)
scRNSTD = np.std(list_real_scoresFNa)

from tabulate import tabulate
print(tabulate([ ['Val Acc.',mx_P,mx_PMean,mx_PSTD,mx_N,mx_NMean,mx_NSTD ],
               ['Test Acc.', scP, scPMean,scPSTD, scN, scNMean,scNSTD],
               ['Real Data Test Acc.', scRP, scRPMean,scRPSTD, scRN, scRNMean,scRNSTD]],
                headers=['TBN Maks', 'TBN Ort','TBN Std', 'UYNB Maks','UYNB Ort','UYNB Std'], tablefmt='orgtbl'))


# In[ ]:


# LETS ADD SIGNIFICANCE TTESTS TO THE RESULTS. BTEK

from scipy.stats import ttest_ind
print('T-test for Test Scores:',ttest_ind(list_scoresFNa, list_scoresPa))
print('T-test for Real Scores:',ttest_ind(list_real_scoresFNa, list_real_scoresPa))



# In[ ]:



print("\n",[l.name  for l in msFN.layers])

print("\n",[l.name  for l in msP.layers])

# # Evaluation and Attention Plots

# In[ ]:


FNSpeechModel = Model(inputs=msFN.input,
                                 outputs=[msFN.get_layer('output').output, 
                                          msFN.get_layer('bidirectional_2').output,
                                          msFN.get_layer('focus-att').output,
                                          msFN.get_layer('dot_1').output,
                                          msFN.get_layer('attSoftmax').output,
                                          msFN.get_layer('dot_2').output])
PSpeechModel = Model(inputs=msP.input,
                                 outputs=[msP.get_layer('output').output, 
                                          msP.get_layer('bidirectional_4').output,                                
                                          msP.get_layer('dense_1').output,
                                          msP.get_layer('dot_3').output,
                                          msP.get_layer('attSoftmax').output,
                                          msP.get_layer('dot_4').output,
                                          msP.get_layer('lambda_1').output])






# In[ ]:


ith = 15
audios, classes = data[6][100:120], data[7][100:120]
print(classes)


# In[ ]:


#8 - on, 13 - one, 7 - right
idAudio =8
print(classes[idAudio])


# In[ ]:


outs = FNSpeechModel.predict(audios)
outsP = PSpeechModel.predict(audios)


# In[ ]:


print(np.argmax(outs[0],axis=1))
print(classes)


# In[ ]:


matplotlib.rcParams.update({'font.size': 22})
imgHeight = 2*2
Fsize = 12


plt.figure(figsize=(Fsize+2,imgHeight))
plt.title('Mel İzgesi', fontsize=30)
plt.ylabel('Büyüklük', fontsize=30)
plt.xlabel('Zaman', fontsize=30)
plt.pcolormesh(audios[idAudio,:,:,0].T)
plt.colorbar()


# plot attention out of softmax
plt.figure(figsize=(Fsize,imgHeight*2))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.plot(outs[4][idAudio],'b-*', linewidth=2)
ax1.plot(outsP[4][idAudio],'r-', linewidth=2)
ax1.set_ylabel('Büyüklük', fontsize=24)
ax1.legend(['uybn (focus)','tbn (dense)'],loc=0)

ax2.plot(np.log(outs[4][idAudio]+1e-20),'b-*', linewidth=2)
ax2.plot(np.log(outsP[4][idAudio]+1e-20),'r-', linewidth=2)
plt.legend(['uybn (focus)','tbn (dense)'],loc=0)
plt.ylabel('Log (Büyüklük)', fontsize=24)
plt.xlabel('Zaman', fontsize=30)
#plt.savefig('picrawWave.png', dpi = 400)
plt.show()

# plot attention for five different inptıs
plt.figure(figsize=(Fsize,imgHeight*2))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.plot((np.array(outs[4])[0,:]).T,'b-*', linewidth=2)
ax1.plot((np.array(outsP[4])[0,:]).T,'r--', linewidth=2)

ax1.plot((np.array(outs[4])[1:5,:]).T,'b-*', linewidth=2)
ax1.plot((np.array(outsP[4])[1:5,:]).T,'r--', linewidth=2)
ax1.set_ylabel('Büyüklük', fontsize=24)
ax1.legend(['uybn (focus)','tbn (dense)'],loc=0)

ax2.plot((np.log(np.array(outs[4])[0:5,:]).T+1e-20),'b-*', linewidth=2,alpha=0.5)
ax2.plot((np.log(np.array(outsP[4])[0:5,:]).T+1e-20),'r--', linewidth=2,alpha=0.5)
plt.ylabel('Log (Büyüklük)', fontsize=24)
plt.xlabel('Zaman', fontsize=30)
#plt.savefig('picrawWave.png', dpi = 400)
plt.show()

# plot bidirectional outputs
plt.figure(figsize=(Fsize,imgHeight))
plt.pcolormesh(outs[1][idAudio])
plt.colorbar()
plt.title('İkiyönlü ÖzYinemeli Katman Çıktısı', fontsize=30)
plt.ylabel('Sıklık', fontsize=30)
plt.xlabel('Zaman', fontsize=30)

# plot bidirectional outputs
plt.figure(figsize=(Fsize,imgHeight))
plt.pcolormesh(outsP[1][idAudio])
plt.colorbar()
plt.title('İkiyönlü ÖzYinemeli Katman Çıktısı', fontsize=30)
plt.pcolormesh(outsP[1][idAudio])
plt.plot([0,128],[65,65],'w--', linewidth=3)
plt.legend(['tbn (dense)'],loc=0)
plt.ylabel('Sıklık', fontsize=30)
plt.xlabel('Zaman', fontsize=30)

#plt.title('Paper bidir output', fontsize=30)

#plt.ylabel('Frequency', fontsize=30)
#plt.xlabel('Time', fontsize=30)
# plot queries directly out of Focus and dense neurons
plt.figure(figsize=(Fsize,imgHeight*2))
plt.plot(outs[2][idAudio],'b-*', linewidth=2)
plt.plot(outsP[2][idAudio],'r-', linewidth=2)
plt.legend(['uybn (focus)','tbn (dense)'],loc=0)
plt.title('Sorgu', fontsize=30)
plt.ylabel('Sıklık', fontsize=30)
plt.xlabel('Zaman', fontsize=30)


plt.figure(figsize=(Fsize,imgHeight*2))
plt.plot(outs[3][idAudio],'b-*', linewidth=2,alpha=0.5)
plt.plot(outsP[3][idAudio],'r-', linewidth=2,alpha=0.5)
plt.legend(['uybn (focus)','tbn (dense)'],loc=0)
plt.title('Dot-1 sorgu-' , fontsize=30)
plt.ylabel('Sıklık', fontsize=30)
plt.xlabel('Zaman', fontsize=30)
plt.show()

plt.figure(figsize=(Fsize,imgHeight*2))
plt.plot(outs[5][idAudio],'b-*', linewidth=2,alpha=0.5)
plt.plot(outsP[5][idAudio],'r-', linewidth=2,alpha=0.5)
plt.legend(['uybn (focus)','tbn (dense)'],loc=0)
plt.title('Öznitelik Vektörü', fontsize=30)
plt.ylabel('Sıklık', fontsize=30)
plt.xlabel('Zaman', fontsize=30)
plt.show()


# In[ ]:

x_test = data[4]
y_test = data[5]
y_predFN = FNSpeechModel.predict(x_test, verbose=1)
y_predP = PSpeechModel.predict(x_test, verbose=1)


# In[ ]:


from sklearn.metrics import confusion_matrix
import audioUtils
cmFN = confusion_matrix(y_test, np.argmax(y_predFN[0],1))
cmP = confusion_matrix(y_test, np.argmax(y_predP[0],1))


# In[ ]:


set(y_test)


# In[ ]:


#35word, v2
classes = ['nine', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
           'zero', 'one', 'two', 'three', 'four', 'five', 'six', 
           'seven',  'eight', 'backward', 'bed', 'bird', 'cat', 'dog',
           'follow', 'forward', 'happy', 'house', 'learn', 'marvin', 'sheila', 'tree',
           'visual', 'wow']


# In[ ]:


#35word, v1
classes=['nine', 'yes',  'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
         'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',  'eight', 
         'bed', 'bird', 'cat', 'dog', 'happy', 'house', 
         'marvin', 'sheila', 'tree', 'wow']


# In[ ]:


#20cmd
classes=['unknown', 'nine', 'yes',  'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
         'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',  'eight']


# In[ ]:


audioUtils.plot_confusion_matrix(cmP,classes, normalize=True)
audioUtils.plot_confusion_matrix(cmFN,classes, normalize=True)


# In[ ]:





# In[ ]:




