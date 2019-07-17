"""
A unified trainer that contains: SVM, Naive Bayes, Random Forest as classifiers
Dataset: ISEAR, EmoSet, TEC, CBET
Created by Chenyang Huang, Dec. 2018
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from data.data_loader import cbet_data, cbet_data_other
from utils.split_k_fold import KFoldsSplitter
from sklearn.metrics import classification_report
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from models.sa_lstm_classifier import AttentionLSTMClassifier
from torch.utils.data import Dataset, DataLoader
from utils.early_stopping import EarlyStopping
import numpy as np
import pickle as pkl
import copy
from tqdm import tqdm
from sklearn.metrics import *
from utils.seq2emo_metric import get_metrics
from utils.tokenizer import BPembTokenizer, GloveTokenizer
from utils.tweet_processor import tweet_process
from copy import deepcopy

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('-dataset', default='median', type=str,
                    help="dataset selection: cbet, emoset, isear, tec")
parser.add_argument('-bs', default=16, type=int)

parser.add_argument('-lr', default=5e-4, type=float)

parser.add_argument('-patience', default=3, type=int)

parser.add_argument('-attention', default='attentive', type=str,
                    help="the attention models: bert, attentive, attention")

parser.add_argument('-tokenizer', default='glove', type=str,
                    help="tokenizer that include embedding, supporting bpemb/glove")

parser.add_argument('-embpath', default='/remote/eureka1/chuang8/glove.840B.300d.txt', type=str,
                    help="tokenizer that include embedding, supporting bpemb/glove")

parser.add_argument('-vocabsize', default=50000, type=int,
                    help="vocab size")

opt = parser.parse_args()

# hyper-parameters
NUM_FOLDS = 10
EMBEDDING_DIM = 300
PAD_LEN = 50
MIN_LEN_DATA = 3
BATCH_SIZE = opt.bs
CLIPS = 0.888  # ref. I Ching, 750 BC
HIDDEN_DIM = 700
VOCAB_SIZE = opt.vocabsize
LEARNING_RATE = opt.lr
PATIENCE = opt.patience
USE_ATT = False


# SELECT DATASET
if opt.dataset == 'cbet':
    X, y, EMO_LIST, NUM_EMO = cbet_data('data/CBET.csv', remove_stop_words=False)
elif opt.dataset == 'small':
    X, y, EMO_LIST, NUM_EMO = cbet_data_other('small', remove_stop_words=False)
elif opt.dataset == 'median':
    X, y, EMO_LIST, NUM_EMO = cbet_data_other('median', remove_stop_words=False)
else:
    raise Exception('Dataset not recognized :(')


EMOS = EMO_LIST
EMOS_DIC = dict(zip(EMOS, range(len(EMOS))))

if opt.tokenizer == 'bpemb':
    tokenizer = BPembTokenizer(VOCAB_SIZE, EMBEDDING_DIM)
elif opt.tokenizer == 'glove':
    tokenizer = GloveTokenizer()
else:
    raise Exception('Tokenizer option is not recognized :(')


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

    model = AttentionLSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, vocab_size,
                                    num_labels, BATCH_SIZE, att_mode=opt.attention, soft_last=False)

    model.load_embedding(tokenizer.get_embeddings())
    # multi-GPU
    # model = nn.DataParallel(model)
    model.cuda()

    loss_criterion = nn.CrossEntropyLoss()  #

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
            y_pred = model(data.cuda(), seq_len)
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
                y_pred = model(_data.cuda(), _seq_len)
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

    with open(f'lstm_{opt.dataset}_model.pt', 'bw') as f:
        torch.save(model.state_dict(), f)

    pred_list = []
    model.eval()
    for _, (_data, _seq_len, _label) in enumerate(test_loader):
        with torch.no_grad():
            y_pred = model(_data.cuda(), _seq_len)
            pred_list.append(y_pred.data.cpu().numpy())  # x[np.where( x > 3.0 )]
            del y_pred

    pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)

    return pred_list


def main():
    global VOCAB_SIZE

    print('Building GloVe tokenizer')
    tokenizer.build_tokenizer(X)
    VOCAB_SIZE = tokenizer.get_vocab_size()

    tokenizer.build_embedding(opt.embpath, save_pkl=True, dataset_name=opt.dataset)

    with open(f'lstm_{opt.dataset}_tokenizer.pkl', 'bw') as f:
        pkl.dump(tokenizer, f)

    kfs = KFoldsSplitter(X, y, NUM_FOLDS, stratified=True)
    X_test, y_test = kfs.get_test()

    for k in range(NUM_FOLDS):
        print("Starting", k + 1, 'fold')
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

        y_pred = model(seq.cuda(), seq_len)

        response = ''
        y_pred = y_pred[0].detach().cpu().numpy()
        y_pred = softmax(y_pred)
        for emo, prob in zip(label_cols, y_pred):
            if prob > 0:
                response += emo + ":" + str(prob) + '\n'

        print(response)

#  main() #used to train and save the pretrained model
interactive_inference()