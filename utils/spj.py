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

def lstm_cell(hidden_dim, p_dropout):
        lstm = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim,state_is_tuple=True)
        lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=p_dropout, output_keep_prob=p_dropout)
        return lstm

class Config(object):
    num_c3d_features = 500
    num_proposals = 10
    num_classes = 10999
    num_steps = 30
    batch_size = 25
    hidden_dim = 512
    num_layers = 2
    eps = 1e-10
    model_name = 'num_c3d_features=%d_num_proposals=%d_num_classes=%d_num_steps=%d_batch_size=%d_hidden_dim=%d_layers=%d_eps=%d' % (num_c3d_features, num_proposals, num_classes, num_steps, batch_size, hidden_dim, num_layers, eps)

class SPJ(object):
    
    def __init__(self, config):
        self.config = config
        
        # Placeholders
        self._H=tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.batch_size,self.config.num_c3d_features,self.config.num_proposals],
            name="H"
        )
        self._Ipast=tf.placeholder(
            dtype=tf.float32, 
            shape=[self.config.batch_size, self.config.num_proposals, self.config.num_proposals], 
            name="Ipast"
        )
        self._Ifuture=tf.placeholder(
            dtype=tf.float32, 
            shape=[self.config.batch_size,self.config.num_proposals,self.config.num_proposals], 
            name="Ifuture"
        )
        self._x=tf.placeholder(
            dtype=tf.int32,
            shape=[self.config.batch_size,self.config.num_proposals,None], 
            name="x"
        )
        self._y=tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.batch_size,self.config.num_proposals,None,self.config.num_classes],
            name="y")
        self._keep_prob=tf.placeholder(
            dtype=tf.int32,shape=()
        )
        self._reg=tf.placeholder(
            dtype=tf.float32,shape=()
        )
        
        # Parameters
        Wa = tf.get_variable(
            name="Wa",
            shape=[self.config.num_c3d_features,self.config.num_c3d_features],
            initializer=tf.contrib.layers.xavier_initializer(seed=1)
        )
        ba = tf.get_variable(
            name="ba", 
            shape=[self.config.num_c3d_features, 1], 
            initializer = tf.zeros_initializer()
        )

        # Begin Attention Module
        # -----------------------
        H2 = tf.transpose(self._H,perm=[1,2,0])
        H3 = tf.reshape(H2,shape=[self.config.num_c3d_features,self.config.num_proposals*self.config.batch_size])
        W = tf.matmul(Wa, H3) + ba 
        W = tf.reshape(W, shape=[self.config.num_c3d_features,self.config.num_proposals, self.config.batch_size]) 
        W = tf.transpose(W,perm=[2,1,0])       # shape = [batch_size, num_proposals, num_c3d_features]
        A = tf.matmul(W,self._H)               # shape = [batch_size, num_proposals, num_proposals]
        Apast =   tf.multiply(A,self._Ipast)
        Afuture = tf.multiply(A,self._Ifuture)
        Zpast =   tf.reduce_sum(self._Ipast,   axis=2) + self.config.eps
        Zfuture = tf.reduce_sum(self._Ifuture, axis=2) + self.config.eps
        Zpast2 =   tf.reshape(Zpast,   shape=[self.config.batch_size, 1, self.config.num_proposals])
        Zfuture2 = tf.reshape(Zfuture, shape=[self.config.batch_size, 1, self.config.num_proposals])
        Hpast = tf.matmul(self._H,tf.transpose(Apast,perm=[0,2,1])) / Zpast2
        Hfuture = tf.matmul(self._H,tf.transpose(Afuture,perm=[0,2,1])) / Zfuture2
        Hout = tf.concat([Hpast, self._H, Hfuture], 1) # shape = [batch_size, 3*num_c3d_features, num_proposals]
        # End Attention Module
        # -----------------------

        
        # Begin Captioning Module
        # -----------------------
        Hout2 = tf.transpose(Hout, perm=[0,2,1]) # shape = [batch_size, num_proposals, num_c3d_features*3]
        Hout3 = tf.reshape(Hout2, [-1, 3*self.config.num_c3d_features]) # shape = [batch_size*num_proposals, 3*num_c3d_features]
        feature_inputs = tf.expand_dims(Hout3,1)

        ## Removed: FC Layer from num_c3d_features to hidden_dim, feature_inputs.shape=[-1, 1, 512]
        #feature_inputs = tf.expand_dims(tf.layers.dense(inputs=Hout,units=self.config.hidden_dim,activation=tf.nn.relu),1) 

        # Trainable Word Embeddings, embedding_inputs.shape=[-1, None, 512]
        embeddings = tf.get_variable('embedding_matrix', [self.config.num_classes, self.config.hidden_dim])
        x2 = tf.reshape(self._x,[self.config.batch_size*self.config.num_proposals,-1]) 
        embedding_inputs = tf.nn.embedding_lookup(
            params = embeddings, 
            ids = x2
        )

        # LSTM Layer
        #print(x2.get_shape().as_list())
        n = tf.shape(x2)[1]
        feature_inputs = tf.tile(input=feature_inputs,multiples=[1,n,1]) # 
        lstm_inputs = tf.concat(values=[feature_inputs, embedding_inputs],axis=2) # [batchsize*30,50,512+1500]
        lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.config.hidden_dim, self._keep_prob) for _ in range(self.config.num_layers)])
        initial_state = lstm.zero_state(self.config.batch_size*self.config.num_proposals, tf.float32) 
        lstm_outputs, final_state = tf.nn.dynamic_rnn(
            cell=lstm,
            inputs=lstm_inputs,
            initial_state=initial_state
        ) # lstm_outputs: [batchsize*30,50,512]
        logits = tf.layers.dense(inputs=tf.reshape(lstm_outputs,[-1,self.config.hidden_dim]),units=self.config.num_classes)
        predictions = tf.argmax(logits,1)
        # End Captioning Module
        # -----------------------

        # Predictions
        self.logits = tf.reshape(logits, [self.config.batch_size,self.config.num_proposals,-1,self.config.num_classes])
        self._predictions = tf.reshape(predictions, [self.config.batch_size,self.config.num_proposals,-1])

        # loss
        #loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits,[-1,self.config.num_classes]), labels=tf.reshape(self._y,[-1])))
        #self.loss = loss
        
        # loss function
        softmax = tf.nn.softmax(logits,axis=1)
        cross_entropy = tf.reshape(self._y,shape=[-1,self.config.num_classes])*tf.log(softmax)
        normalizer = (self.config.batch_size*self.config.num_proposals*self.config.num_steps)
        loss = -tf.reduce_sum(tf.reduce_sum(cross_entropy,axis=1))/(normalizer)                                             
        # Regularization
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #loss = my_normal_loss + reg_constant * sum(reg_losses)
        self._loss = loss
        
    
    def caption_generation(self,sess,minibatch_H,minibatch_Ipast,minibatch_Ifuture,minibatch_Xcaptions,minibatch_Ycaptions):
        minibatch_Ycaptions = id_2_one_hot_void_padding(minibatch_Ycaptions, self.config.num_classes, void_dim=0)
        batch_size = minibatch_H.shape[0]
        sent_pred = np.ones([batch_size,self.config.num_proposals,1])*2 # <START>
        prev_caption = np.zeros(minibatch_Xcaptions.shape)
        ind = [i for i in range(self.config.num_classes)]
        
        while sent_pred.shape[2] < self.config.num_steps: 
            prev_caption[:,:,:sent_pred.shape[2]] = sent_pred
            idx_next_pred = sent_pred.shape[2]+1
            logits = sess.run([self.logits], 
                      feed_dict={self._H: minibatch_H, self._Ipast: minibatch_Ipast, 
                                 self._Ifuture: minibatch_Ifuture, self._x: prev_caption, 
                                 self._y: minibatch_Ycaptions,
                                 self._keep_prob: 1.0})
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
    
    def generate_caption_2(self, session, H, Ipast, Ifuture, labels):        
        assert (H.shape[0] == self.config.batch_size),"batch sizes do not match!"
        x = np.ones([self.config.batch_size, self.config.num_proposals, 1])*2 # b'<sta>': 2
        y = np.ones([self.config.batch_size, self.config.num_proposals,1,self.config.num_classes])
        while (x.shape[2] - 1) < self.config.num_steps:
            feed = {self._H: H,
                    self._Ipast: Ipast,
                    self._Ifuture: Ifuture,
                    self._x: x,
                    self._y: y,
                    self._keep_prob: 1.0}
            predictions = session.run(self._predictions, feed_dict=feed)
            next_x = predictions[:,:,-1]
            next_x = np.expand_dims(next_x, axis=2)
            x = np.concatenate((x,next_x),axis=2)
            y = np.ones([self.config.batch_size, self.config.num_proposals,x.shape[2],self.config.num_classes])
        return predictions, labels
    

    
    

    