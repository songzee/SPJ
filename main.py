# As usual, a bit of setup
import time
import numpy as np
import matplotlib.pyplot as plt
import LOUPE.WILLOW.loupe as lp
import tensorflow as tf
import h5py
import pandas as pd
import csv
import copy
import math
from utils.data_utils import temporal_indicator
from utils.data_utils import temporal_pooling
from module import *


#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2


# format
train_data = pd.read_csv("train_extreme_small.csv")
train_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
train_data["duration"] = train_data["duration"].astype('float32')
train_data["t_init"], train_data["t_end"] = train_data["timestamps"].str.split(", ", 1).str
train_data["t_init"] = train_data["t_init"].str.strip("[")
train_data["t_end"] = train_data["t_end"].str.strip("]")
train_data["t_init"] = train_data["t_init"].astype('float32')
train_data["t_end"] = train_data["t_end"].astype('float32')
train_data = train_data.drop('timestamps', 1)

# pool
filename = "/home/songzeli/Data/sub_activitynet_v1-3.c3d.hdf5"
video_feature_representation = h5py.File(filename, 'r')
train_ids = train_data['id'].unique()
f_inits = []
f_ends = []
max_pooled_representations = []
max_proposals = 0
padded_proposals = np.zeros((99,500,30))
padded_framestamps = -1*np.ones((99,2,30))
for v,video_id in enumerate(train_ids):
    #print("video id: ", video_id)
    temp = train_data[train_data['id']==video_id].reset_index()
    C3D_features = video_feature_representation["v_QOlSCBRmfWY"]['c3d_features'].value
    
    if max_proposals < temp.shape[0]:
        max_proposals = temp.shape[0]
    
    for i in range(temp.shape[0]):
        
        # get time info
        duration = temp["duration"][i]
        t_init = temp["t_init"][i]
        t_end = temp["t_end"][i]
        num_frames = C3D_features.shape[0]
        
        # compute start and end frame
        f_init = int(round((t_init/duration)*num_frames))
        f_end = int(round((t_end/duration)*num_frames))
        #print("f_init: ", f_init, "t_init: ", t_init)
        #print("f_end: ", f_end, "t_end: ", t_end)
        
        # get max pool
        if f_init <= f_end:
            max_pooled_rep = temporal_pooling(C3D_features[f_init:f_end],"max")
        else:
            max_pooled_rep = temporal_pooling(C3D_features[f_end:f_init],"max")
        
        # append info
        f_inits.append(f_init)
        f_ends.append(f_end)
        max_pooled_representations.append(max_pooled_rep)
        padded_proposals[v,:,i] = max_pooled_rep
        padded_framestamps[v,0,i] = f_init
        padded_framestamps[v,1,i] = f_end


f_inits = np.array(f_inits)
f_inits = pd.DataFrame({'f_init': f_inits})
f_ends = np.array(f_ends) 
f_ends = pd.DataFrame({'f_end': f_ends})

max_pooled_representations = np.array(max_pooled_representations)
C3D_feature_column_names = ["h" + str(i) for i in range(max_pooled_representations.shape[1])] 
max_pooled_representations = pd.DataFrame(max_pooled_representations, columns=C3D_feature_column_names)

train_data = pd.concat([train_data, f_inits, f_ends, max_pooled_representations], axis=1)


train_data.to_pickle("train_data")

print("number of examples: ", train_ids.shape[0])
print("train_data.shape: ", train_data.shape)
print("padded_proposals.shape: ", padded_proposals.shape)
print("padded_framestamps.shape: ", padded_framestamps.shape) 


#Changed by Songze
train_voc = pd.read_csv("train_all.csv")
train_voc.rename( columns={'Unnamed: 0':'index'}, inplace=True )
train_voc["duration"] = train_voc["duration"].astype('float32')
train_voc["t_init"], train_voc["t_end"] = train_voc["timestamps"].str.split(", ", 1).str
train_voc["t_init"] = train_voc["t_init"].str.strip("[")
train_voc["t_end"] = train_voc["t_end"].str.strip("]")
train_voc["t_init"] = train_voc["t_init"].astype('float32')
train_voc["t_end"] = train_voc["t_end"].astype('float32')
train_voc = train_voc.drop('timestamps', 1)
from utils.data_utils import export_vocabulary
export_vocabulary(train_voc)
#Changed by Songze


#Changed by Songze
df = pd.read_csv('vocabulary.csv')
voc = df["Unnamed: 0"].tolist()
vocabulary = []
for word in voc:
    if word.isalpha():
        vocabulary.append(word)
len(vocabulary)
#Changed by Songze

import sys
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_STA = b"<sta>"
_END = b"<end>"
_START_VOCAB = [_PAD,_UNK,_STA,_END]
PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3

# word_path = './vocabulary.csv'
emb_dim = 512
vocab_size = len(vocabulary)

emb_matrix,word2id,id2word = get_wordvector(emb_dim,vocab_size,vocabulary)


import re
#padding for sentences
pad_len = 50

all_padded_sentences = np.zeros((99,pad_len,30))
for v,video_id in enumerate(train_ids):
    temp = train_data[train_data['id']==video_id].reset_index()
    for i in range(temp.shape[0]):
        words,ids = sentence_to_token_ids(temp['sentences'][i][:-1],word2id)
        ids_pad = padded(ids,pad_len)
        all_padded_sentences[v,:,i] = ids_pad

                   
all_padded_sentences_2 = np.zeros((99,pad_len+1,30))
for v,video_id in enumerate(train_ids):
    temp = train_data[train_data['id']==video_id].reset_index()
    for i in range(temp.shape[0]):
        words,ids = sentence_to_token_ids(temp['sentences'][i][:-1],word2id)
        ids_pad = padded(ids,pad_len+1)
        all_padded_sentences_2[v,:,i] = ids_pad

all_padded_sentences_id = np.array(all_padded_sentences).astype(int)
        
print("all_padded_sentences_2.shape: ", all_padded_sentences_2.shape)


H_train = padded_proposals.astype(np.float32)
framestamps = padded_framestamps
Ipast = temporal_indicator(framestamps, mode="past")
Ipast_train = Ipast.astype(np.float32)
Ifuture = temporal_indicator(framestamps, mode="future")
Ifuture_train = Ifuture.astype(np.float32)
emb_matrix, word2id, id2word = get_wordvector(emb_dim,vocab_size,vocabulary) #changed by Songze
sentence_ids = all_padded_sentences_id
Ycaptions = copy.deepcopy(all_padded_sentences_2) # holds i
Xcaptions = copy.deepcopy(all_padded_sentences)
Xcaptions = Xcaptions.astype(np.int32)
Ycaptions = Ycaptions.astype(np.int32)
Xcaptions_train = np.transpose(Xcaptions,axes=(0,2,1))
Ycaptions_train = np.transpose(Ycaptions,axes=(0,2,1))


tf.reset_default_graph()

num_c3d_features = 500
num_classes = len(word2id)
hidden_dim = 512
num_steps = 50
num_proposals = 30
num_layers = 2
batch_size = 9

tf.reset_default_graph()
# create placeholders
H = tf.placeholder(tf.float32,shape=[batch_size, num_c3d_features, num_proposals], name="H")
Ipast = tf.placeholder(tf.float32, shape=[batch_size, num_proposals, num_proposals], name="Ipast")
Ifuture = tf.placeholder(tf.float32, shape=[batch_size, num_proposals, num_proposals], name="Ifuture")
x = tf.placeholder(tf.int32, [batch_size, num_proposals, num_steps], name="x")
y = tf.placeholder(tf.int32, [batch_size, num_proposals, num_steps+1], name="y")

# forward pass
Hout = attention_module(H, Ipast, Ifuture, num_proposals, num_c3d_features, num_steps, batch_size)
predictions, logits = language_module(Hout, x, num_classes, hidden_dim, num_steps, num_proposals, num_layers, batch_size)
cap_loss = caption_cost(y, logits, num_classes, num_proposals,num_steps,batch_size)


# check forward pass
#with tf.Session() as sess:
#    # Run the initialization
#    sess.run(init)
#    minibatch_cost = sess.run([cap_loss], feed_dict={H: H_train[0:batch_size], Ipast: Ipast_train[0:batch_size], Ifuture: Ifuture_train[0:batch_size], x: Xcaptions_train[0:batch_size], y: Ycaptions_train[0:batch_size]})


# Baseline Test Case
H = padded_proposals.astype(np.float32)
framestamps = padded_framestamps
Ipast = temporal_indicator(framestamps, mode="past")
Ipast = Ipast.astype(np.float32)
Ifuture = temporal_indicator(framestamps, mode="future")
Ifuture = Ifuture.astype(np.float32)
emb_matrix, word2id, id2word = get_wordvector(emb_dim,vocab_size,vocabulary) #changed by Songze
sentence_ids = all_padded_sentences_id
Ycaptions = copy.deepcopy(all_padded_sentences_2) # holds i
Xcaptions = copy.deepcopy(all_padded_sentences)

Xcaptions = Xcaptions.astype(np.int32)
Ycaptions = Ycaptions.astype(np.int32)

Xcaptions = np.transpose(Xcaptions,axes=(0,2,1))
Ycaptions = np.transpose(Ycaptions,axes=(0,2,1))

print(Ycaptions.dtype)
print("H.shape: ", H.shape)
print("framestamps.shape: ", framestamps.shape)
print("Ipast.shape: ", Ipast.shape)
print("Ifuture.shape: ", Ifuture.shape)
print("Ycaptions.shape: ", Ycaptions.shape)
print("Xcaptions.shape: ", Xcaptions.shape)

model(H, Ipast, Ifuture, Ycaptions, Xcaptions)
