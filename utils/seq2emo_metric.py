from sklearn import metrics
import numpy as np


def conver_to_binary_any(output):
    """ Any positive prediction
    Conver output of the shape (B * num_emo) -> (B * num_emo)
    :param output: output of Seq2Emo model (B * num_emo * num_emo*2)
    :return: convert_output (B * num_emo)
    """
    B, num_emo = output.shape
    converted_output = []
    for row in output:
        # row = row.reshape((B, -1))
        positive_pos = (row[row % 2 == 1]-1)/2
        positive_pos = positive_pos.astype(np.int)
        converted = np.asarray([0]*num_emo)
        converted[positive_pos] = 1
        converted_output.append(converted)

    return np.asarray(converted_output)


def conver_to_binary(output):
    """   ONLY TRUE WHEN THE POSITION IS ALSO CORRECT
    Conver output of the shape (B * num_emo) -> (B * num_emo)
    :param output: output of Seq2Emo model (B * num_emo * num_emo*2)
    :return: convert_output (B * num_emo)
    """
    B, num_emo = output.shape
    pos_mask = list(range(num_emo))
    pos_mask = [2 * x + 1 for x in pos_mask]
    pos_mask = np.asarray(pos_mask)
    converted_output = []
    for row in output:
        converted = np.asarray([0]*num_emo)
        converted[row == pos_mask] = 1
        converted_output.append(converted)

    return np.asarray(converted_output)


def get_metrics(y, y_pre):
    hamming_loss = metrics.hamming_loss(y, y_pre)
    macro_f1 = metrics.f1_score(y, y_pre, average='macro')
    macro_precision = metrics.precision_score(y, y_pre, average='macro')
    macro_recall = metrics.recall_score(y, y_pre, average='macro')
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    micro_precision = metrics.precision_score(y, y_pre, average='micro')
    micro_recall = metrics.recall_score(y, y_pre, average='micro')
    return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


def get_multi_metrics(y, y_pre):
    multi_condition = np.sum(y, axis=1) > 1
    y = y[multi_condition]
    y_pre = y_pre[multi_condition]

    hamming_loss = metrics.hamming_loss(y, y_pre)
    macro_f1 = metrics.f1_score(y, y_pre, average='macro')
    macro_precision = metrics.precision_score(y, y_pre, average='macro')
    macro_recall = metrics.recall_score(y, y_pre, average='macro')
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    micro_precision = metrics.precision_score(y, y_pre, average='micro')
    micro_recall = metrics.recall_score(y, y_pre, average='micro')
    return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


# conver_to_binary = conver_to_binary_only
