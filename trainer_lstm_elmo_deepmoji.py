"""
A unified trainer that contains: SVM, Naive Bayes, Random Forest as classifiers
Dataset: ISEAR, EmoSet, TEC, CBET
Created by Chenyang Huang, Dec. 2018
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from data.data_loader import cbet_data, cbet_data_other
from utils.split_k_fold import KFoldsSplitter
from sklearn.metrics import classification_report
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from models.sa_lstm_classifier_elmo_deepmoji import AttentionLSTMClassifier
from torch.utils.data import Dataset, DataLoader
from utils.early_stopping import EarlyStopping
import numpy as np
import pickle as pkl
import copy
from tqdm import tqdm
from sklearn.metrics import *
from utils.seq2emo_metric import get_metrics
from utils.tokenizer import BPembTokenizer, GloveTokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
# emoji transfer
import json
from utils.focalloss import FocalLoss
from copy import deepcopy
from utils.tweet_processor import tweet_process

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('-dataset', default='median', type=str,
                    help="dataset selection: cbet")

parser.add_argument('-lr', default=5e-4, type=float)

parser.add_argument('-patience', default=3, type=int)

parser.add_argument('-dim', default=800, type=int)

parser.add_argument('-attention', default='attentive', type=str,
                    help="the attention models: bert, attentive, attention")

parser.add_argument('-useglove', default='True', type=str,
                    help="to use a glove the embedding will be a combination of ")

parser.add_argument('-embpath', default='/remote/eureka1/chuang8/glove.840B.300d.txt', type=str,
                    help="tokenizer that include embedding, supporting bpemb/glove")

parser.add_argument('-elmo', default='origin', type=str,
                    help="origin/origin55b")

parser.add_argument('-vocabsize', default=50000, type=int,
                    help="vocabulary size")

parser.add_argument('-addlinear', default='True', type=str,
                    help="add linear layer after deepmoji")

parser.add_argument('-bs', default=32, type=int,
                    help="batch size")

parser.add_argument('-savepkl', default='True', type=str,
                    help="batch size")

parser.add_argument('-maxpool', default='False', type=str,
                    help="max pooling over deepmoji")

parser.add_argument('-loss', default='ce', type=str,
                    help="ce/focal")

parser.add_argument('-focal', default=2, type=int,
                    help="gamma value for focal loss")

opt = parser.parse_args()

print('attention type:', opt.attention,
      '\ndataset:', opt.dataset,
      '\nuse_glove:', opt.useglove,
      '\nvocab size:', opt.vocabsize,
      '\nbatch size:', opt.bs)

# hyper-parameters
NUM_FOLDS = 10
EMBEDDING_DIM = 300
PAD_LEN = 50
MIN_LEN_DATA = 3
BATCH_SIZE = opt.bs
CLIPS = 0.888  # ref. I Ching, 750 BC
HIDDEN_DIM = opt.dim
VOCAB_SIZE = opt.vocabsize
LEARNING_RATE = opt.lr
PATIENCE = opt.patience
USE_ATT = False
SAVE_PKL = True
ADD_LINEAR = True
MAX_POOLING = False

if opt.useglove == 'True':
    USE_GLOVE = True
elif opt.useglove == 'False':
    USE_GLOVE = False
else:
    raise Exception('useglove arg not recognized')

if opt.addlinear == 'True':
    ADD_LINEAR = True
elif opt.addlinear == 'False':
    ADD_LINEAR = False
else:
    raise Exception('addlinear arg not recognized')

if opt.maxpool == 'True':
    MAX_POOLING = True
elif opt.maxpool == 'False':
    MAX_POOLING = False
else:
    raise Exception('maxpool arg not recognized')

if opt.savepkl == 'True':
    SAVE_PKL = True
elif opt.savepkl == 'False':
    SAVE_PKL = False
else:
    raise Exception('savepkl arg not recognized')

if opt.attention not in ["bert", "attentive", "attention", "None"]:
    raise Exception('attention mode not recognized')


# SELECT DATASET
if opt.dataset == 'cbet':
    X, y, EMO_LIST, NUM_EMO = cbet_data('data/CBET.csv', remove_stop_words=False)
elif opt.dataset == 'small':
    X, y, EMO_LIST, NUM_EMO = cbet_data_other('small', remove_stop_words=False)
elif opt.dataset == 'median':
    X, y, EMO_LIST, NUM_EMO = cbet_data_other('median', remove_stop_words=False)
else:
    raise Exception('Dataset not recognized :(')


if opt.elmo == 'origin':
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elif opt.elmo == 'origin55b':
    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
else:
    raise Exception('elmo model not recognized')

elmo = Elmo(options_file, weight_file, 2, dropout=0).cuda()
elmo.eval()

EMOS = EMO_LIST
EMOS_DIC = dict(zip(EMOS, range(len(EMOS))))

tokenizer = GloveTokenizer()

# deepmoji
print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, PAD_LEN)

print('Loading model from {}.'.format(PRETRAINED_PATH))
emoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)
emoji_model.eval()

class EmotionDataLoader(Dataset):
    def __init__(self, X, y, pad_len, max_size=None):
        self.source = []
        self.source_len = []
        self.target = y
        self.pad_len = pad_len
        self.read_data(X, y)

    def read_data(self, X, y):
        for src in X:
            src = tokenizer.encode_ids(src)
            if len(src) < self.pad_len:
                src_len = len(src)
                src = src + [0] * (self.pad_len - len(src))
            else:
                src = src[:self.pad_len]
                src_len = self.pad_len

            self.source_len.append(src_len)
            self.source.append(src)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.LongTensor(self.source[idx]), \
               torch.LongTensor([self.source_len[idx]]), \
               torch.LongTensor([self.target[idx]])


def train(X_train, y_train, X_dev, y_dev, X_test, y_test):
    num_labels = NUM_EMO

    vocab_size = VOCAB_SIZE

    print('NUM of VOCAB' + str(vocab_size))
    train_data = EmotionDataLoader(X_train, y_train, PAD_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    dev_data = EmotionDataLoader(X_dev, y_dev, PAD_LEN)
    dev_loader = DataLoader(dev_data, batch_size=int(BATCH_SIZE/3)+2, shuffle=False)

    test_data = EmotionDataLoader(X_test, y_test, PAD_LEN)
    test_loader = DataLoader(test_data, batch_size=int(BATCH_SIZE/3)+2, shuffle=False)

    model = AttentionLSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, num_labels, BATCH_SIZE, att_mode=opt.attention,
                                    soft_last=False, use_glove=USE_GLOVE, add_linear=ADD_LINEAR, max_pool=MAX_POOLING)

    if USE_GLOVE:
        model.load_embedding(tokenizer.get_embeddings())
    # multi-GPU
    # model = nn.DataParallel(model)
    model.cuda()

    if opt.loss == 'ce':
        loss_criterion = nn.CrossEntropyLoss()  #
        print('Using ce loss')
    elif opt.loss == 'focal':
        loss_criterion = FocalLoss(gamma=opt.focal, reduce=True)
        print('Using focal loss, gamma=', opt.focal)
    else:
        raise Exception('loss option not recognised')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    es = EarlyStopping(patience=PATIENCE)

    old_model = None
    for epoch in range(1, 300):
        print('Epoch: ' + str(epoch) + '===================================')
        train_loss = 0
        model.train()
        for i, (data, seq_len, label) in tqdm(enumerate(train_loader),
                                              total=len(train_data)/BATCH_SIZE):
            optimizer.zero_grad()

            data_text = [tokenizer.decode_ids(x) for x in data]
            with torch.no_grad():
                character_ids = batch_to_ids(data_text).cuda()
                elmo_emb = elmo(character_ids)['elmo_representations']
                elmo_emb = (elmo_emb[0] + elmo_emb[1])/2 # avg of two layers

                emoji_tokenized, _, _ = st.tokenize_sentences([' '.join(x) for x in data_text])
                emoji_encoding = emoji_model(torch.LongTensor(emoji_tokenized.astype(np.int32)))

            y_pred = model(data.cuda(), seq_len, elmo_emb, emoji_encoding.cuda())
            loss = loss_criterion(y_pred, label.view(-1).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPS)
            optimizer.step()
            train_loss += loss.data.cpu().numpy() * data.shape[0]
            del y_pred, loss

        test_loss = 0
        model.eval()
        for _, (_data, _seq_len, _label) in enumerate(dev_loader):
            with torch.no_grad():

                data_text = [tokenizer.decode_ids(x) for x in _data]
                character_ids = batch_to_ids(data_text).cuda()
                elmo_emb = elmo(character_ids)['elmo_representations']
                elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers

                emoji_tokenized, _, _ = st.tokenize_sentences([' '.join(x) for x in data_text])
                emoji_encoding = emoji_model(torch.LongTensor(emoji_tokenized.astype(np.int32)))

                y_pred = model(_data.cuda(), _seq_len, elmo_emb, emoji_encoding.cuda())
                loss = loss_criterion(y_pred, _label.view(-1).cuda())
                test_loss += loss.data.cpu().numpy() * _data.shape[0]
                del y_pred, loss

        print("Train Loss: " + str(train_loss / len(train_data)) + \
              " Evaluation: " + str(test_loss / len(dev_data)))

        if es.step(test_loss):  # overfitting
            del model
            print('overfitting, loading best model ...')
            model = old_model
            break
        else:
            if es.is_best():
                if old_model is not None:
                    del old_model
                print('saving best model ...')
                old_model = deepcopy(model)
            else:
                print('not best model, ignoring ...')
                if old_model is None:
                    old_model = deepcopy(model)

    with open(f'lstm_elmo_deepmoji_{opt.dataset}_model.pt', 'bw') as f:
        torch.save(model.state_dict(), f)

    pred_list = []
    model.eval()
    for _, (_data, _seq_len, _label) in enumerate(test_loader):
        with torch.no_grad():
            data_text = [tokenizer.decode_ids(x) for x in _data]
            character_ids = batch_to_ids(data_text).cuda()
            elmo_emb = elmo(character_ids)['elmo_representations']
            elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers

            emoji_tokenized, _, _ = st.tokenize_sentences([' '.join(x) for x in data_text])
            emoji_encoding = emoji_model(torch.LongTensor(emoji_tokenized.astype(np.int32)))

            y_pred = model(_data.cuda(), _seq_len, elmo_emb, emoji_encoding.cuda())
            pred_list.append(y_pred.data.cpu().numpy())  # x[np.where( x > 3.0 )]
            del y_pred

    pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)

    return pred_list


def main():
    global VOCAB_SIZE

    print('Building GloVe tokenizer')
    tokenizer.build_tokenizer(X)
    VOCAB_SIZE = tokenizer.get_vocab_size()
    if USE_GLOVE:
        print('Building GloVe embedding')
        tokenizer.build_embedding(opt.embpath, save_pkl=SAVE_PKL, dataset_name=opt.dataset)

    with open(f'lstm_elmo_deepmoji_{opt.dataset}_tokenizer.pkl', 'bw') as f:
        pkl.dump(tokenizer, f)

    kfs = KFoldsSplitter(X, y, NUM_FOLDS, stratified=True)
    X_test, y_test = kfs.get_test()

    for k in range(NUM_FOLDS):
        print("Starting", k+1, 'fold')
        X_train, y_train, X_dev, y_dev = kfs.next_fold()

        y_test_pred = train(X_train, y_train, X_dev, y_dev, X_test, y_test)

        kfs.add_result(y_test_pred)

        break

    y_test_pred_mj = kfs.get_voting_result()

    print("Final voting results")
    print(classification_report(y_test, y_test_pred_mj, target_names=EMO_LIST, digits=4))


def interactive_inference(model_token=''):
    with open(f'lstm_{model_token}{opt.dataset}_tokenizer.pkl', 'br') as f:
        tokenizer = pkl.load(f)

    def encode_seq(src):
        src = tokenizer.encode_ids(src)
        if len(src) < PAD_LEN:
            src_len = len(src)
            src = src + [0] * (PAD_LEN - len(src))
        else:
            src = src[:PAD_LEN]
            src_len = PAD_LEN

        return torch.LongTensor(src).unsqueeze(0), \
               torch.LongTensor([src_len]).unsqueeze(0)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    model = AttentionLSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, tokenizer.get_vocab_size(),
                                    NUM_EMO, BATCH_SIZE, att_mode=opt.attention, soft_last=False)
    # multi-GPU
    # model = nn.DataParallel(model)
    with open(f'lstm_{model_token}{opt.dataset}_model.pt', 'br') as f:
        model.load_state_dict(torch.load(f))

    model.cuda()
    label_cols = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']

    while True:
        print('type "end" to terminate >>> ')
        text = input()
        if text.strip().lower() == 'end':
            break
        text = tweet_process(text)

        seq, seq_len = encode_seq(text)

        data_text = [tokenizer.decode_ids(x) for x in seq]
        character_ids = batch_to_ids(data_text).cuda()
        elmo_emb = elmo(character_ids)['elmo_representations']
        elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers

        emoji_tokenized, _, _ = st.tokenize_sentences([' '.join(x) for x in data_text])
        emoji_encoding = emoji_model(torch.LongTensor(emoji_tokenized.astype(np.int32)))

        y_pred = model(seq.cuda(), seq_len, elmo_emb, emoji_encoding.cuda())

        response = ''
        y_pred = y_pred[0].detach().cpu().numpy()
        y_pred = softmax(y_pred)
        for emo, prob in zip(label_cols, y_pred):
            if prob > 0:
                response += emo + ":" + str(prob) + '\n'

        print(response)

main()
interactive_inference('elmo_deepmoji_')
