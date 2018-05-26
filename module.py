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
import re
_PAD = b"<pad>"
_UNK = b"<unk>"
_STA = b"<sta>"
_END = b"<end>"
_START_VOCAB = [_PAD,_UNK,_STA,_END]
PAD_ID = 0                                                    
UNK_ID = 1                                                    
STA_ID = 2                                                    
END_ID = 3       

def get_wordvector(emb_dim,vocab_size,vocabulary):
    """Reads from original word lib file and returns embedding matrix and
    mappings from words to word ids.

    Returns:
      emb_matrix: Numpy array shape (len(vocabulary), word_dim) containing word embeddings
        (plus PAD and UNK embeddings in first 4 rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """
    
    vocabulary = _START_VOCAB + vocabulary

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), emb_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize all the tokens
    emb_matrix[:, :] = np.random.randn(vocab_size + len(_START_VOCAB), emb_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in vocabulary:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1


    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w.lower(), UNK_ID) for w in tokens]
    return tokens, ids


def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(lambda x: len(x), token_batch) if batch_pad == 0 else batch_pad
    res = [STA_ID]+token_batch+[END_ID]+[PAD_ID] * (maxlen - len(token_batch)-2)

    return res


def attention_module(H, Ipast, Ifuture, num_proposals, num_c3d_features, num_steps, batch_size):
    """
    Implements the attention module: see https://cs.stanford.edu/people/ranjaykrishna/densevid/
    
    Arguments:
    H -- input dataset placeholder, of shape = [None, N, K] and dtype "float"
    Ipast -- placeholder for the indicators of past, shape = [None, K, K] and dtype "float"
    Ifuture == placeholder for the indicators of future, shape = [None, K, K] and dtype "float"
    parameters -- python dictionary containing your parameters "Wa", "ba", sapes [N,N] and [N,1] respectively

    Returns:
    Hout -- concatenated output (hpast, h, hfuture), shape = [batch_size, 3*num_c3d_features, num_proposals]
    """

    print("H dtype: ", H.dtype)
    print("Ipast dtype: ", Ipast.dtype)
    print("Ifuture dtype: ", Ifuture.dtype)
    #print("x dtype: ", x.dtype)
    #print("y dtype: ", y.dtype)
    
    # Retrieve the parameters from the dictionary "parameters" 
    Wa = tf.get_variable("Wa", [num_c3d_features,num_c3d_features], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    ba = tf.get_variable("ba", [num_c3d_features,1], initializer = tf.zeros_initializer())

    # forward pass
    W = tf.transpose(tf.tensordot(Wa,tf.transpose(H,perm=[1,2,0]),axes=[[1], [0]]),perm=[2,0,1]) + ba # shape: [None,num_proposals,num_proposals]
    A = tf.matmul(tf.transpose(W,perm=[0,2,1]),H) # shape: [None,num_proposals,num_proposals]
    A_flat = tf.reshape(A, [-1, num_proposals*num_proposals]) # shape: [None,num_proposals*num_proposals]

    # future features
    Ifuture_flat = tf.reshape(Ifuture, [-1, num_proposals*num_proposals]) # shape: [None,K*K]
    Afuture = tf.reshape(tf.multiply(Ifuture_flat,A_flat),[-1,num_proposals,num_proposals]) # shape: [None,K,K]
    Zfuture = tf.reduce_sum(Ifuture,axis=2) # shape: [None,num_proposals]
    Hfuture = tf.transpose(tf.transpose(tf.matmul(H,tf.transpose(Afuture,perm=[0,2,1])),perm=[1,0,2])/Zfuture,perm=[1,0,2]) # shape: [None,num_c3d_features,num_proposals]

    # past features
    Ipast_flat = tf.reshape(Ipast, [-1, num_proposals*num_proposals]) # shape: [None,num_proposals*num_proposals]
    Apast = tf.reshape(tf.multiply(Ipast_flat,A_flat),[-1,num_proposals,num_proposals]) # shape: [None,num_proposals,num_proposals]
    Zpast = tf.reduce_sum(Ipast,axis=2) # shape: [None,num_proposals]
    Hpast = tf.transpose(tf.transpose(tf.matmul(H,tf.transpose(Apast,perm=[0,2,1])),perm=[1,0,2])/Zfuture,perm=[1,0,2]) # shape: [None,num_c3d_features,num_proposals]

    # stacked features
    Hout = tf.concat([Hpast, H, Hfuture], 1)

    print("Hfuture shape: ", Hfuture.get_shape().as_list())
    print("W shape: ", W.get_shape().as_list())
    print("A shape: ", A.get_shape().as_list())
    print("A_flat shape: ", A_flat.get_shape().as_list())
    print("Ifuture_flat shape: ", Ifuture_flat.get_shape().as_list())
    print("Zfuture: ", Zfuture.get_shape().as_list())
    print("Hfuture: ", Hfuture.get_shape().as_list())
    print("Hpast: ", Hfuture.get_shape().as_list())
    print("Hout: ", Hout.get_shape().as_list())
    
    return Hout

def language_module(Hout, x, num_classes, hidden_dim, num_steps, num_proposals, num_layers, batch_size):
    '''
    Inputs: 
      number of classes
      hidden_dim = number units in lstm and word embedding
      num_steps, length of captions
      num_steps
      num_layers 
      batch_size
    '''
    
    Hout = tf.transpose(Hout, perm=[0,2,1])
    Hout = tf.reshape(Hout, [-1, 1500])
    
    # create placeholder
    #x_captions = tf.placeholder(tf.int32, [batch_size, num_proposals, num_steps], name="x_captions")
    
    
    feature_inputs = tf.expand_dims(tf.layers.dense(inputs=Hout,units=hidden_dim,activation=tf.nn.relu),1)
    print("feature_inputs.shape: ",feature_inputs.get_shape().as_list())

    embeddings = tf.get_variable('embedding_matrix', [num_classes, hidden_dim])
    embedding_inputs = tf.nn.embedding_lookup(embeddings, tf.reshape(x,[-1,num_steps]))
    print("embedding_inputs.shape: ",embedding_inputs.get_shape().as_list())
                                              
    lstm_inputs = tf.concat(values=[feature_inputs, embedding_inputs],axis=1)
    print("all_inputs.shape: ", lstm_inputs.get_shape().as_list())                                      
    
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim,state_is_tuple=True)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
    initial_state = lstm_cells.zero_state(batch_size*num_proposals, tf.float32)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cells,inputs=lstm_inputs,initial_state=initial_state)
    print("lstm_outputs.shape: ", lstm_outputs.get_shape().as_list())                                         
                                              
    logits = tf.layers.dense(inputs=tf.reshape(lstm_outputs,[-1,hidden_dim]),units=num_classes)
                                         
    predictions = tf.argmax(logits,1)
    predictions = tf.reshape(predictions, [batch_size,num_proposals,num_steps+1])
    print("predictions.shape: ", predictions.get_shape().as_list())
    
    logits = tf.reshape(logits, [batch_size,num_proposals,num_steps+1,num_classes])
    print("logits.shape: ", logits.get_shape().as_list())
                                                                            
    return predictions, logits

def caption_cost(y, logits, num_classes, num_proposals,num_steps,batch_size):
    print("y_captions.shape: ", y.get_shape().as_list())
    print()
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.reshape(logits,[-1,num_classes]), 
            labels=tf.reshape(y,[-1])
        )
    )
    return loss

def model(H_train, Ipast_train, Ifuture_train, Ycaptions_train, Xcaptions_train, learning_rate = 0.001, num_epochs = 3, minibatch_size = 9, print_cost = True):
    """
    Implements a tensorflow neural network: C3D->DAPS->ATTENTION->CAPTIONING
    
    Arguments:
    H_train -- training set, of shape = [n_train,num_c3d_features,num_proposals]
    Y_train -- caption labels, of shape = [n_train,num_proposals,num_steps+1]
    H_test -- training set, of shape = [n_test,num_c3d_features,num_proposals]
    Y_test -- caption labels, of shape = [n_test,num_proposals,num_steps+1]
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    batch_size = minibatch_size
    
    # to be able to rerun the model without overwriting tf variables
    tf.reset_default_graph()    
    
    # to keep consistent results
    tf.set_random_seed(1)                             
    seed = 3                                         
    
    # size values
    (n_train,num_c3d_features,num_proposals) = H_train.shape                        
    (_,_,num_steps) = Xcaptions_train.shape
    num_classes = 400002
    num_layers = 2
    hidden_dim = 512
    
    print("n_train ", n_train)
    print("num_c3d_features ", num_c3d_features)
    print("num_proposals: ", num_proposals)
    print("num_steps: ", num_steps)
    
    # to keep track of costs
    costs = []  
    
    # create placeholders
    H = tf.placeholder(tf.float32,shape=[batch_size, num_c3d_features, num_proposals], name="H")
    Ipast = tf.placeholder(tf.float32, shape=[batch_size, num_proposals, num_proposals], name="Ipast")
    Ifuture = tf.placeholder(tf.float32, shape=[batch_size, num_proposals, num_proposals], name="Ifuture")
    x = tf.placeholder(tf.int32, [batch_size, num_proposals, num_steps], name="x")
    y = tf.placeholder(tf.int32, [batch_size, num_proposals, num_steps+1], name="y")
    
    # attention module
    #attention_module(K,N, batch_size)
    Hout = attention_module(
        H,
        Ipast,
        Ifuture,
        num_proposals, 
        num_c3d_features,
        num_steps,
        batch_size
    )
    
    # language module
    #language_module(Hout, num_classes, hidden_dim, num_steps, num_proposals, num_layers, batch_size)
    predictions, logits = language_module(
        Hout, 
        x,
        num_classes, 
        hidden_dim, 
        num_steps, 
        num_proposals, 
        num_layers, 
        batch_size
    )
    
    # cost
    # caption_cost(logits, num_classes, num_proposals,num_steps,batch_size)
    cap_loss = caption_cost(
        y,
        logits, 
        num_classes, 
        num_proposals, 
        num_steps, 
        batch_size
    )
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cap_loss)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            print("epoch: ", epoch)

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(n_train / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(H_train, Ipast_train, Ifuture_train, Ycaptions_train, Xcaptions_train, minibatch_size, seed = 0)

            for counter,minibatch in enumerate(minibatches):
                print("counter: ", counter)

                # Select a minibatch
                (minibatch_H, minibatch_Ipast, minibatch_Ifuture, minibatch_Ycaptions, minibatch_Xcaptions) = minibatch
                print("minibatch_H.shape: ", minibatch_H.shape)
                print("minibatch_Ipast.shape: ", minibatch_Ipast.shape)
                print("minibatch_Ifuture.shape: ", minibatch_Ifuture.shape)
                print("minibatch_Ycaptions.shape: ", minibatch_Ycaptions.shape)
                print("minibatch_Xcaptions.shape: ", minibatch_Xcaptions.shape)
                
                print(type(minibatch_H))
                print(type(minibatch_Ipast))
                print(type(minibatch_Ifuture))
                print(type(minibatch_Ycaptions))
                print(type(minibatch_Xcaptions))
                
                # The line that runs the graph on a minibatch.
                _ , minibatch_cost = sess.run([optimizer, cap_loss], feed_dict={H: minibatch_H, Ipast: minibatch_Ipast, Ifuture: minibatch_Ifuture, x: minibatch_Xcaptions, y: minibatch_Ycaptions})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


def random_mini_batches(H, Ipast, Ifuture, Ycaptions, Xcaptions, mini_batch_size = 9, seed = 0):
    """
    Creates a list of random minibatches from (H, Ipast, Ifuture, Ycaptions, Xcaptions)
    
    Arguments:
    H -- training set, of shape = [n_train,num_c3d_features,num_proposals]
    Y -- caption labels, of shape = [n_train,num_proposals,num_steps+1]
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    """
    
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
     """
    
    m = H.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (H, Ipast, Ifuture, Ycaptions, Xcaptions)
    permutation = list(np.random.permutation(m))
    shuffled_H = H[permutation]
    shuffled_Ipast = Ipast[permutation]
    shuffled_Ifuture = Ifuture[permutation]
    shuffled_Ycaptions = Ycaptions[permutation]
    shuffled_Xcaptions = Xcaptions[permutation]
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_H = shuffled_H[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Ipast = shuffled_Ipast[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Ifuture = shuffled_Ifuture[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Ycaptions = shuffled_Ycaptions[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Xcaptions = shuffled_Xcaptions[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_H, mini_batch_Ipast, mini_batch_Ifuture, mini_batch_Ycaptions, mini_batch_Xcaptions)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_H = shuffled_H[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Ipast = shuffled_Ipast[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Ifuture = shuffled_Ifuture[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Ycaptions = shuffled_Ycaptions[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Xcaptions = shuffled_Xcaptions[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_H, mini_batch_Ipast, mini_batch_Ifuture, mini_batch_Ycaptions, mini_batch_Xcaptions)
        mini_batches.append(mini_batch)
    
    return mini_batches

