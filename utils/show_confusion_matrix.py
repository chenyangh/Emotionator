import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import pandas as pd

emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']
reverse_index = [0, 7, 1, 2, 4, 5, 3, 6, 8]
forward_index = [0, 2, 3, 6, 4, 5, 7, 1, 8]
new_emo = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'love', 'thankfulness', 'guilt']


def confusion_matrix(file_name):
    def pred_count(preds):
        my_dict = {}
        for i in range(9):
            my_dict[i] = 0
        for item in preds:
            my_dict[forward_index[item]] += 1

        return my_dict
    df = pd.read_csv(file_name)
    cm = np.zeros([len(emotions), len(emotions)])
    for i, emo in enumerate(new_emo):
        preds = df[emo]
        my_dict = pred_count(preds)
        cm[i, :] = np.asarray([my_dict[x] for x in range(len(emotions))], dtype=np.int64)
    return cm


def plot_confusion_matrix(name, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, new_emo, rotation=90)
    plt.yticks(tick_marks, new_emo)
    plt.tick_params(labelsize=16)
    ax = plt.gca()
    for i in range(6):
        ax.get_xticklabels()[i].set_color("red")
        ax.get_yticklabels()[i].set_color("red")

    for i in range(6, 9):
        ax.get_xticklabels()[i].set_color("blue")
        ax.get_yticklabels()[i].set_color("blue")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.gcf().subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.savefig('heat_maps/' + name + '.pdf', format='pdf')
    print(name, 'done!')

np.set_printoptions(precision=2)

name_list = ['simple_beam1_bar_', 'simple_beam2_bar_', 'simple_beam3_bar_', 'simple_beam4_bar_', 'simple_beam5_bar_',
             'simple_beam1_foo_', 'simple_beam2_foo_', 'simple_beam3_foo_', 'simple_beam4_foo_', 'simple_beam5_foo_',
             'persona_beam1_', 'persona_beam2_', 'persona_beam3_', 'persona_beam4_', 'persona_beam5_']


name_list = ['simple_beam2_bar_',
             'simple_beam2_foo_',
             'persona_beam2_']

name_list = ['simple_beam2_bar_',
             'simple_beam2_foo_',
             'persona_beam2_',
             'decoder_att_beam2_',
             'decoder_transformation_beam2_',
             'decoder_vocablayer_beam2_',
             'simple_start_beam_2_bar_']


name_list = ['decoder_att_beam2_', 'decoder_transformation_beam2_', 'decoder_vocablayer_beam2_', 'simple_start_beam_2_bar_']



for name in name_list:
    file_name = 'results2/' + name + 'result.csv'
    class_names = emotions
    plt.figure()
    cnf_matrix = confusion_matrix(file_name)
    plot_confusion_matrix(name, cnf_matrix, classes=class_names, normalize=True,
                          title='')