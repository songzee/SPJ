import tensorflow as tf
import h5py
import pandas as pd
import csv
import copy
import math
from utils.data_utils import temporal_indicator
from utils.data_utils import temporal_pooling
from utils.data_utils import export_vocabulary
import sys
import re

class Config(object):
    num_c3d_features = 500
    num_proposals = 30
    num_classes = 10194
    num_steps = 50
    hidden_dim = 512
    num_layers = 2
    eps = 1e-10
    model_name = 'num_c3d_features=%d_num_proposals=%d_num_classes=%d_num_steps=%d_hidden_dim=%d_layers=%d_eps=%d' % (num_c3d_features, num_proposals, num_classes, num_steps, hidden_dim, num_layers, eps)

class SPJ(object):
    
    def __init__(self, config):
        self.config = config
        
        
        # placeholders
        self._batchsize = tf.placeholder(tf.int32,shape=())
        self._H=tf.placeholder(tf.float32,shape=[None,self.config.num_c3d_features,self.config.num_proposals],name="H")
        self._Ipast=tf.placeholder(tf.float32, shape=[None, self.config.num_proposals, self.config.num_proposals], name="Ipast")
        self._Ifuture=tf.placeholder(tf.float32, shape=[None,self.config.num_proposals,self.config.num_proposals], name="Ifuture")
        self._x=tf.placeholder(tf.int32,shape=[None,self.config.num_proposals,self.config.num_steps], name="x")
        self._y=tf.placeholder(tf.int32,shape=[None,self.config.num_proposals,self.config.num_steps+1],name="y")

        # Attention Parameters
        Wa = tf.get_variable("Wa",[self.config.num_c3d_features,self.config.num_c3d_features],initializer=tf.contrib.layers.xavier_initializer(seed=1))
        ba = tf.get_variable("ba", [self.config.num_c3d_features, 1], initializer = tf.zeros_initializer())

        # Forward Pass
        W = tf.transpose(tf.tensordot(Wa,tf.transpose(self._H,perm=[1,2,0]),axes=[[1], [0]]),perm=[2,0,1]) + ba # shape: [None,num_proposals,num_proposals]
        A = tf.matmul(tf.transpose(W,perm=[0,2,1]),self._H) # shape: [None,num_proposals,num_proposals]
        A_flat = tf.reshape(A, [-1, self.config.num_proposals*self.config.num_proposals]) 

        # Future Features
        Ifuture_flat = tf.reshape(self._Ifuture, [-1, self.config.num_proposals*self.config.num_proposals]) # shape: [None,K*K]
        Afuture = tf.reshape(tf.multiply(Ifuture_flat,A_flat),[-1, self.config.num_proposals, self.config.num_proposals]) # shape: [None,K,K]
        Zfuture = tf.reduce_sum(self._Ifuture,axis=2) + self.config.eps # shape: [None,num_proposals]
        Hfuture = tf.transpose(tf.transpose(tf.matmul(self._H,tf.transpose(Afuture,perm=[0,2,1])),perm=[1,0,2])/Zfuture,perm=[1,0,2]) # shape: [None,num_c3d_features,num_proposals]

        # Past Features
        Ipast_flat = tf.reshape(self._Ipast, [-1, self.config.num_proposals*self.config.num_proposals]) # shape: [None,num_proposals*num_proposals]
        Apast = tf.reshape(tf.multiply(Ipast_flat,A_flat),[-1,self.config.num_proposals, self.config.num_proposals]) # shape: [None,num_proposals,num_proposals]
        Zpast = tf.reduce_sum(self._Ipast, axis=2) + self.config.eps # shape: [None,num_proposals]
        Hpast = tf.transpose(tf.transpose(tf.matmul(self._H,tf.transpose(Apast,perm=[0,2,1])),perm=[1,0,2])/Zfuture,perm=[1,0,2]) # shape: [None,num_c3d_features,num_proposals]

        # Stacked Features
        Hout = tf.concat([Hpast, self._H, Hfuture], 1)

        # Reshape Hout
        Hout = tf.transpose(Hout, perm=[0,2,1])
        Hout = tf.reshape(Hout, [-1, 3*self.config.num_c3d_features]) 

        # FC Layer from num_c3d_features to hidden_dim
        feature_inputs = tf.expand_dims(tf.layers.dense(inputs=Hout,units=self.config.hidden_dim,activation=tf.nn.relu),1) 

        # Trainable Word Embeddings
        embeddings = tf.get_variable('embedding_matrix', [self.config.num_classes, self.config.hidden_dim])
        embedding_inputs = tf.nn.embedding_lookup(embeddings, tf.reshape(self._x,[-1,self.config.num_steps]))

        # LSTM Layer
        lstm_inputs = tf.concat(values=[feature_inputs, embedding_inputs],axis=1) 
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_dim,state_is_tuple=True)
        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers, state_is_tuple=True)
        initial_state = lstm_cells.zero_state(self._batchsize*self.config.num_proposals, tf.float32)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cells,inputs=lstm_inputs,initial_state=initial_state)                                                                          
        logits = tf.layers.dense(inputs=tf.reshape(lstm_outputs,[-1,self.config.hidden_dim]),units=self.config.num_classes)
        predictions = tf.argmax(logits,1)
        predictions = tf.reshape(predictions, [-1,self.config.num_proposals,self.config.num_steps+1])
        logits = tf.reshape(logits, [-1,self.config.num_proposals,self.config.num_steps+1,self.config.num_classes])


        # Predictions
        self.logits = logits
        self._predictions = predictions

        # loss
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits,[-1,self.config.num_classes]), labels=tf.reshape(self._y,[-1])))
        self.loss = loss

    



