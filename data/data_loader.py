import pandas as pd
import numpy as np
from nltk.corpus import stopwords


def cbet_data(file_path='data/CBET.csv', remove_stop_words=True, get_text=True, preprocess=True, multi=False, vector=False):
    NUM_CLASS = 9
    emo_list = ["anger", "fear", "joy", "love", "sadness", "surprise", "thankfulness", "disgust", "guilt"]
    stop_words = set(stopwords.words('english'))

    label = []
    train_text = []
    df = pd.read_csv(file_path)
    for i, row in df.iterrows():
        if get_text:
            from utils.tweet_processor import tweet_process
            text = row['text']
            if preprocess:
                text = tweet_process(text)
            if remove_stop_words:
                text = ' '.join([x for x in text.split() if x not in stop_words])
            train_text.append(text)

        emo_one_hot = row[emo_list]
        emo_one_hot = np.asarray(emo_one_hot)
        if not multi:
            if sum(emo_one_hot) != 1:
                continue
            emo_idx = np.argmax(emo_one_hot)
        else:
            if not vector:
                emo_idx = np.where(emo_one_hot == 1)[0].tolist()
            else:
                emo_idx = emo_one_hot
        label.append(emo_idx)

    return train_text, label, emo_list, NUM_CLASS


def cbet_data_other(mode='small', remove_stop_words=True, get_text=True, preprocess=True, vector=False):
    """
    :param mode: Small or median
    :param remove_stop_words:
    :param get_text:
    :param preprocess:
    :param multi:
    :param vector:
    :return:
    """
    assert mode in ['small', 'median']
    NUM_CLASS = 9
    emo_list = ["anger", "fear", "joy", "love", "sadness", "surprise", "thankfulness", "disgust", "guilt"]
    stop_words = set(stopwords.words('english'))

    label = []
    train_text = []

    if mode == 'small':
        file_path = 'data/CBET-single-small.txt'
    else:
        file_path = 'data/CBET-single-medium.txt'

    file_reader = open(file_path, 'r')
    for row in file_reader.readlines():
        tokens = row.strip().split('\t\t')
        if get_text:
            from utils.tweet_processor import tweet_process
            text = tokens[0]
            if preprocess:
                text = tweet_process(text)
            if remove_stop_words:
                text = ' '.join([x for x in text.split() if x not in stop_words])
            train_text.append(text)

        emo = int(tokens[1])
        if not vector:
            label.append(emo)
        else:
            emo_one_hot = np.zeros(NUM_CLASS)
            emo_one_hot[emo] = 1
            label.append(emo_one_hot)

    return train_text, label, emo_list, NUM_CLASS


