import tensorflow as tf
import h5py
import pandas as pd
import csv
import copy
import math
from utils.data_utils import *
import sys
import re
import numpy as np
from utils.data_utils import sample

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
        #self._y=tf.placeholder(tf.int32,shape=[None,self.config.num_proposals,self.config.num_steps+1],name="y")
        self._y=tf.placeholder(tf.float32,shape=[None,self.config.num_proposals,self.config.num_steps+1,self.config.num_classes],name="y")

        # Begin Attention Module
        # -----------------------
        
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
        
        # End Attention Module
        # -----------------------

        
        
        # Begin Captioning Module
        # -----------------------
        
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
        
        # End Captioning Module
        # -----------------------
        
        
        #logits = tf.reshape(logits, [-1,self.config.num_proposals,self.config.num_steps+1,self.config.num_classes])


        # Predictions
        self.logits = tf.reshape(logits, [-1,self.config.num_proposals,self.config.num_steps+1,self.config.num_classes])
        self._predictions = tf.reshape(predictions, [-1,self.config.num_proposals,self.config.num_steps+1])

        # loss
        #loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits,[-1,self.config.num_classes]), labels=tf.reshape(self._y,[-1])))
        #self.loss = loss
        
        # loss function
        softmax = tf.nn.softmax(logits,axis=1)
        cross_entropy = tf.reshape(self._y,shape=[-1,self.config.num_classes])*tf.log(softmax)
        self.loss = loss = -tf.reduce_sum(tf.reduce_sum(cross_entropy,axis=1))
        
    
    def caption_generation(self,sess,minibatch_H,minibatch_Ipast,minibatch_Ifuture,minibatch_Xcaptions,minibatch_Ycaptions):
        minibatch_Ycaptions = id_2_one_hot_void_padding(minibatch_Ycaptions, self.config.num_classes, void_dim=0)
        batch_size = minibatch_H.shape[0]
        sent_pred = np.ones([batch_size,self.config.num_proposals,1])*2 # <START>
        prev_caption = np.zeros(minibatch_Xcaptions.shape)
        
        while sent_pred.shape[2] < self.config.num_steps: 
            prev_caption[:,:,:sent_pred.shape[2]] = sent_pred
            idx_next_pred = sent_pred.shape[2]+1
            logits = sess.run([self.logits], 
                      feed_dict={self._H: minibatch_H, self._Ipast: minibatch_Ipast, 
                                 self._Ifuture: minibatch_Ifuture, self._x: prev_caption, 
                                 self._y: minibatch_Ycaptions,
                                 self._batchsize: minibatch_H.shape[0]})
            logits = logits[0]
            next_logits = logits[:,:,idx_next_pred,:]
            raw_predicted = np.zeros([next_logits.shape[0],next_logits.shape[1],1])
            for batch_idx in range(next_logits.shape[0]):
                for proposal_idx in range(next_logits.shape[1]):
                    idx = sample(next_logits[batch_idx,proposal_idx,:])
#                     print(next_logits[batch_idx,proposal_idx,:])
                    print(idx)
                    raw_predicted[batch_idx,proposal_idx] = idx
            raw_predicted = np.array(raw_predicted)
            sent_pred = np.concatenate([sent_pred, raw_predicted], 2)
            print(sent_pred.shape)
        return sent_pred
    