"""
A generator for reading and serving audio files

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

Remember to use multiprocessing:
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
                    
"""

import numpy as np
import keras

class SpeechGen(keras.utils.Sequence):
    """
    'Generates data for Keras'
    
    list_IDs - list of files that this generator should load
    labels - dictionary of corresponding (integer) category to each file in list_IDs
    
    Expects list_IDs and labels to be of the same length
    """
    def __init__(self, list_IDs, labels, batch_size=64, dim=16000, shuffle=True, mels=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.mels=mels #if mels is true dim must be 3dims, e.g. 125,80,1

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.mels:
            X = np.empty(((self.batch_size,)+ self.dim))
        else:
            X = np.empty((self.batch_size, self.dim))
        
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            #load data from file, saved as numpy array on disk
            curX = np.load(ID)
            
            #normalize
            #invMax = 1/(np.max(np.abs(curX))+1e-3)
            #curX *= invMax   
            # Store class
            y[i] = self.labels[ID]         
            
            if self.mels: # we are working with mfcc
                if curX.shape == self.dim:
                    X[i] = curX
                    continue
                else:
                    print("shapes do not match, we dont expect this!")
                    return None
            else: # we are working on wav files I suppose
                #curX could be bigger or smaller than self.dim
                if curX.shape[0] == self.dim:
                    X[i] = curX
                    #print('Same dim')
                elif curX.shape[0] > self.dim: #bigger
                    #we can choose any position in curX-self.dim
                    randPos = np.random.randint(curX.shape[0]-self.dim)
                    X[i] = curX[randPos:randPos+self.dim]
                    #print('File dim bigger')
                else: #smaller
                    randPos = np.random.randint(self.dim-curX.shape[0])
                    X[i,randPos:randPos+curX.shape[0]] = curX
                    #print('File dim smaller')
            
            

        return X, y
