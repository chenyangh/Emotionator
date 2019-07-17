import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import multiprocessing
import numpy as np
from multiprocessing import Pool
import emoji
import string
printable = set(string.printable)
#
emotion_hashtags = {'anger': ['anger',  'rage', 'pain', 'angry'],
                    'fear': ['fear', 'anxiety', 'horror', 'horrific'],
                    'joy': ['joy', 'happy', 'like', 'happiness', 'smile', 'peace', 'pleased', 'satisfied',
                            'satisfying'],
                    'love': ['love', 'beautiful'],
                    'sadness': ['sadness', 'sad', 'sadness', 'depression', 'depressed', 'alone',
                                'loneliness', 'lonely'],
                    'surprise': ['surprise', 'amazing', 'awesome', 'fascinate', 'fascinating', 'incredible',
                                 'marvelous', 'prodigious', 'shocking', 'stunning', 'surprising', 'unbelievable'],
                    'thankfulness': ['thankfulness', 'thankful', 'gratitude', 'kindness', 'thanks', 'gratefulness',
                                     'grateful'],
                    'disgust': ['disgust', 'disgusting', 'dislike', 'antipathy', 'distaste', 'distasteful', 'hatred',
                                'loathing'],
                    'guilt': ['guilt', 'guilty', 'culpability', 'disgrace', 'indiscretion', 'liability', 'regret',
                              'remorse', 'responsibility', 'shame', 'shameful', 'sin']
                    }

emotion_tags = []
for emo in emotion_hashtags:
    emotion_tags.extend(['#' + x for x in emotion_hashtags[emo]])


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
    #           'emphasis', 'censored'},

    annotate={"repeated", "emphasis", "elongated"},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def remove_tags(s):
    return ' '.join([x for x in s.split() if x not in emotion_tags])


def emotion_detector(tweet):
    tokens = tweet.split()
    emo_found = []
    for token in tokens:
        if token.startswith('#'):
            for emo, tag_list in emotion_hashtags.items():
                for word in tag_list:
                    emo_hashtag = '#' + word
                    if emo_hashtag == token:
                        if emo not in emo_found:
                            emo_found.append(emo)
    return emo_found


num_partitions = 12
num_cores = multiprocessing.cpu_count()


def parallelize_dataframe(df, func):
    part_list = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, part_list))
    pool.close()
    pool.join()
    return df


def process_tweet(s):
    return ' '.join(text_processor.pre_process_doc(remove_tags(s)))


def tweet_process(text):
    text = ' '.join(text_processor.pre_process_doc(remove_tags(text)))
    text = emoji.demojize(text, delimiters=(' ', ' '))
    tokens = text.split()
    ret_list = []
    for token in tokens:
        if len(token) > 3 and '_' in token:
            token = token.replace('_', ' ')

        if token[0] == '<' and token[-1] == '>':
            token = token[1:-1]

        ret_list.append(token)
    text = ' '.join(ret_list)
    return text


def remove_dup_emoji(sent):
    ret = []
    for word in sent.split():
        emo_found = [char for char in word if char in UNICODE_EMOJI]
        if len(emo_found) > 1:
            word = emo_found[0]
        ret.append(word)
    return ' '.join(ret)


def remove_underscope_for_emoji(text):
    tokens = text.split()
    ret_list = []
    for token in tokens:
        if len(token) > 3 and '_' in token:
            token = token.replace('_', ' ')

        if token[0] == '<' and token[-1] == '>':
            token = token[1:-1]

        ret_list.append(token)
    return ' '.join(ret_list)


def only_printable(text):
    """
    Usage Warning, for the sake of efficient, this method did not rejoin the string with space
    Therefore, in the 'processing_pipeline', I put it before 'remove_underscope_for_emoji'
    """

    text = ''.join([x for x in text if x in printable])
    return text


def processing_pipeline(text, remove_hashtag=False):
    text = text.lower().strip()
    if remove_hashtag:
        text = remove_tags(text)
    text = ' '.join(text_processor.pre_process_doc(text))
    # text = only_printable(text)
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = only_printable(text)
    text = remove_underscope_for_emoji(text)
    return text

# print(processing_pipelie('e day હત ા શ ા ર ો ગ મ ા ટ ે હ ો મ ી ય ો પ ે થ ી homeop'))