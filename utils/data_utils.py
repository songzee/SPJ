import numpy as np
import pandas as pd
import csv

def temporal_pooling(features,mode="max"):
    """
    Computes pooling accross frames according to mode
    
    Arguments:
    features -- numpy array shape = (num_frames, num_features)
    mode     -- "max" or "average"
    
    Returns:
    h - shape = (num_features,)
    """
    if mode=="max":
        h = np.max(features,axis=0)  
    elif mode=="average":
        h = np.mean(features,axis=0) 
    return h


def temporal_indicator(framestamps, mode):
    """
    Computes binary matrix to use in attention for past or future
    
    Arguments:
    framestamps -- numpy array shape [batch_size x 2 x K]  
    mode     -- "past" or "future"
    
    Returns:
    Imode - Indicator numpy array mode [batch_size x K x K]
    """
    batch_size, _ , K = framestamps.shape
    frame_ends = framestamps[:,1,:]    # shape: [bach_size,K]
    Imode = np.zeros((K,batch_size,K)) # shape: [K,batch_size,K]
    for k in range(K):
        #if 
        if mode=="past":
            Imode[k] = frame_ends < frame_ends[:,k].reshape(batch_size,1)
        if mode=="future":
            Imode[k] = frame_ends >= frame_ends[:,k].reshape(batch_size,1)
            Imode[k,:,k] = np.zeros((batch_size))
    Imode = np.transpose(Imode,(1,0,2))
    return Imode

def export_vocabulary(train_data):
    caption_words = []
    for caption in train_data["sentences"]:
        caption_words.extend(caption.strip('.').strip(', ').replace("'", "").lower().split(' '))
    print("Total number of words in all captions: ", len(caption_words))
    word_set = set(caption_words)
    vocabulary = list(word_set)
    print("Vocabulary Size (Unique): ", len(vocabulary))
    vocabulary

    csvfile = "vocabulary.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in vocabulary:
            writer.writerow([val])