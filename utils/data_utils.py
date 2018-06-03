import numpy as np
import pandas as pd
import csv
import string
import h5py
import re
import math
from sklearn.utils.extmath import softmax

def temporal_pooling(features,mode="max"):
    """
    Computes pooling accross frames according to mode
    
    Arguments:
    features -- numpy array shape = (num_frames, num_features)
    mode     -- "max" or "average"
    
    Returns:
    h - shape = (num_features,)
    """
    #if features.shape[0] == 1:
    #    return features
    #print(features.shape)
    if features.shape[0] == 0:
        print(features)
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
    delset = string.punctuation
    trans = str.maketrans('', '', string.punctuation)
    for caption in train_data["sentences"]:
#         caption = caption.translate(None,delset)
#         caption_words.extend(caption.strip('.').strip(', ').replace("'", "").lower().split(' '))
#           caption_words.extend(caption.strip('.').strip(', ').strip('"').strip('"').strip(",").replace("'", "").lower().split(' '))
        caption_words.extend(caption.translate(trans).lower().split(' '))
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
                     
def video_preprocess(home_dir, train_file):
    # format
    train_data = pd.read_csv(train_file)
    num_examples = len(train_data['id'].unique())
    train_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    train_data["duration"] = train_data["duration"].astype('float32')
    train_data["t_init"], train_data["t_end"] = train_data["timestamps"].str.split(", ", 1).str
    train_data["t_init"] = train_data["t_init"].str.strip("[")
    train_data["t_end"] = train_data["t_end"].str.strip("]")
    train_data["t_init"] = train_data["t_init"].astype('float32')
    train_data["t_end"] = train_data["t_end"].astype('float32')
    train_data = train_data.drop('timestamps', 1)

    # pool
    filename = home_dir + "/Data/sub_activitynet_v1-3.c3d.hdf5"
    video_feature_representation = h5py.File(filename, 'r')
    train_ids = train_data['id'].unique()
    f_inits = []
    f_ends = []
    max_pooled_representations = []
    max_proposals = 0
    padded_proposals = np.zeros((num_examples,500,30))
    padded_framestamps = -1*np.ones((num_examples,2,30))
    for v,video_id in enumerate(train_ids):
        #print("video id: ", video_id)
        temp = train_data[train_data['id']==video_id].reset_index()
        #C3D_features = video_feature_representation["v_QOlSCBRmfWY"]['c3d_features'].value
        C3D_features = video_feature_representation[video_id]['c3d_features'].value
        #print(video_id)

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
            #print(f_init)
            #print(f_end)
            if f_init <= f_end:
                max_pooled_rep = temporal_pooling(C3D_features[f_init:f_end+1],"max")
            else:
                max_pooled_rep = temporal_pooling(C3D_features[f_end:f_init+1],"max")

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

    #print("number of examples: ", train_ids.shape[0])
    #print("train_data.shape: ", train_data.shape)
    #print("padded_proposals.shape: ", padded_proposals.shape)
    #print("padded_framestamps.shape: ", padded_framestamps.shape) 
    return train_ids,train_data,padded_proposals,padded_framestamps

def caption_preprocess(home_dir):  
    train_voc = pd.read_csv(home_dir + "/SPJ/train_all.csv")
    train_voc.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    train_voc["duration"] = train_voc["duration"].astype('float32')
    train_voc["t_init"], train_voc["t_end"] = train_voc["timestamps"].str.split(", ", 1).str
    train_voc["t_init"] = train_voc["t_init"].str.strip("[")
    train_voc["t_end"] = train_voc["t_end"].str.strip("]")
    train_voc["t_init"] = train_voc["t_init"].astype('float32')
    train_voc["t_end"] = train_voc["t_end"].astype('float32')
    train_voc = train_voc.drop('timestamps', 1)

    export_vocabulary(train_voc)
    df = pd.read_csv(home_dir + '/SPJ/vocabulary.csv')
    header_name = list(df.columns.values)[0]
    voc = df[header_name].tolist()
    vocabulary = []
    for word in voc:
        if word.isalpha():
            vocabulary.append(word)
            
    vocab_size = len(vocabulary)
    return vocabulary,vocab_size

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
    _PAD = b"<pad>"
    _UNK = b"<unk>"
    _STA = b"<sta>"
    _END = b"<end>"
    _START_VOCAB = [_PAD,_UNK,_STA,_END]
    PAD_ID = 0
    UNK_ID = 1
    STA_ID = 2
    END_ID = 3
    
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
#Changed by Songze
def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    _PAD = b"<pad>"
    _UNK = b"<unk>"
    _STA = b"<sta>"
    _END = b"<end>"
    _START_VOCAB = [_PAD,_UNK,_STA,_END]
    PAD_ID = 0
    UNK_ID = 1
    STA_ID = 2
    END_ID = 3
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
    _PAD = b"<pad>"
    _UNK = b"<unk>"
    _STA = b"<sta>"
    _END = b"<end>"
    _START_VOCAB = [_PAD,_UNK,_STA,_END]
    PAD_ID = 0
    UNK_ID = 1
    STA_ID = 2
    END_ID = 3
    maxlen = max(lambda x: len(x), token_batch) if batch_pad == 0 else batch_pad
    res = [STA_ID]+token_batch+[END_ID]+[PAD_ID] * (maxlen - len(token_batch)-2)

    return res

def get_padded_sentences_id(pad_len,train_ids, train_data,word2id):
    _PAD = b"<pad>"
    _UNK = b"<unk>"
    _STA = b"<sta>"
    _END = b"<end>"
    _START_VOCAB = [_PAD,_UNK,_STA,_END]
    PAD_ID = 0
    UNK_ID = 1
    STA_ID = 2
    END_ID = 3
    num_examples = len(train_ids)
    all_padded_sentences = np.zeros((num_examples,pad_len,30))
    for v,video_id in enumerate(train_ids):
        temp = train_data[train_data['id']==video_id].reset_index()
        for i in range(temp.shape[0]):
            words,ids = sentence_to_token_ids(temp['sentences'][i][:-1],word2id)
            ids_pad = padded(ids,pad_len)
            if len(ids_pad) > pad_len:
                ids_pad = ids_pad[:pad_len]
            all_padded_sentences[v,:,i] = ids_pad


    all_padded_sentences_2 = np.zeros((num_examples,pad_len+1,30))
    for v,video_id in enumerate(train_ids):
        temp = train_data[train_data['id']==video_id].reset_index()
        for i in range(temp.shape[0]):
            words,ids = sentence_to_token_ids(temp['sentences'][i][:-1],word2id)
            ids_pad = padded(ids,pad_len+1)
            if len(ids_pad) > pad_len:
                ids_pad = ids_pad[:pad_len+1]
            all_padded_sentences_2[v,:,i] = ids_pad
    all_padded_sentences_id = np.array(all_padded_sentences).astype(int)
            
    return all_padded_sentences,all_padded_sentences_2,all_padded_sentences_id

def random_mini_batches(VideoIds, Framestamps, H, Ipast, Ifuture, Ycaptions, Xcaptions, mini_batch_size = 9, seed = 0):
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
    shuffled_VideoIds = VideoIds[permutation]
    shuffled_Framestamps = Framestamps[permutation]
    shuffled_H = H[permutation]
    shuffled_Ipast = Ipast[permutation]
    shuffled_Ifuture = Ifuture[permutation]
    shuffled_Ycaptions = Ycaptions[permutation]
    shuffled_Xcaptions = Xcaptions[permutation]
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_VideoIds = shuffled_VideoIds[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Framestamps = shuffled_Framestamps[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_H = shuffled_H[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Ipast = shuffled_Ipast[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Ifuture = shuffled_Ifuture[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Ycaptions = shuffled_Ycaptions[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Xcaptions = shuffled_Xcaptions[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_VideoIds, mini_batch_Framestamps, mini_batch_H, mini_batch_Ipast, mini_batch_Ifuture, mini_batch_Ycaptions, mini_batch_Xcaptions)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_VideoIds = shuffled_VideoIds[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Framestamps = shuffled_Framestamps[num_complete_minibatches * mini_batch_size : m]
        mini_batch_H = shuffled_H[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Ipast = shuffled_Ipast[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Ifuture = shuffled_Ifuture[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Ycaptions = shuffled_Ycaptions[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Xcaptions = shuffled_Xcaptions[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_VideoIds, mini_batch_Framestamps, mini_batch_H, mini_batch_Ipast, mini_batch_Ifuture, mini_batch_Ycaptions, mini_batch_Xcaptions)
        mini_batches.append(mini_batch)
    
    return mini_batches

def id_2_one_hot_void_padding(y, num_classes, void_dim=-1):
    '''
    input: 
     y.shape = [batchsize, dim1, dim2], where each element is the class id
     num_classes: number of classes
     void_dim: the class you want to zero out or -1 if not zeroing out
    
    return: 
     one_hot with dimension void_dim set to zero
    '''
    batchsize, dim1, dim2 = y.shape
    y = y.flatten()
    one_hot = np.zeros((batchsize*dim1*dim2,num_classes))
    one_hot[np.arange(len(y)), y] = 1.0
    if void_dim > -1:
        one_hot[:,void_dim] = 0.0
    one_hot = one_hot.reshape((batchsize,dim1,dim2,num_classes))
    return one_hot


def sample(a, temperature=2.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a[a < 0] = 0
    a = np.log(a) / (temperature)
    b = np.exp(a) / (np.sum(np.exp(a)))
    i = 1
    while(sum(b)>1):
        b[-i] = 0
        i +=1
    print(b)
    return np.argmax(np.random.multinomial(1, b, 1))
# def sample(a,ind):
#     a = np.expand_dims(a,axis=0)
#     b = softmax(a)
#     b = np.squeeze(b,axis=0)
#     return np.argmax(np.random.choice(ind,p = b))

def print_pred_and_labels(predictions, labels, videoids, id2word, example=0, proposal=0):
    num_steps = predictions.shape[2]
    print()
    print('{:20.20}'.format('VIDEO ID'),'{:20.20}'.format('PREDICTION'), '{:20.20}'.format('LABEL'))
    print('{:20.20}'.format('--------'), '{:20.20}'.format('-----'), '{:20.20}'.format('-----'))
    for i in range(num_steps):
        print('{:20.20}'.format(str(videoids[example])),'{:20.20}'.format(str(id2word[predictions[example,proposal,i]]) ), '{:20.20}'.format(str(id2word[labels[example,proposal,i]])))
    return 0
