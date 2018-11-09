import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from common import config
import json


def save_model(words, features, filename):
    output_file = open(filename, 'w', encoding='utf-8')

    for word, feature in zip(words, features):
        output_file.write('{} '.format(word))

        for num in feature:
            output_file.write('{} '.format(num))
        output_file.write('\n')

    output_file.close()
    print('result saved to {}'.format(filename))


def save_features(feats, cur_epoch, tb_log_dir, word2idx):
    word_list_file_name = 'eval/ITC/word_list.txt'
    word_list = []
    for idx, line in enumerate(open(word_list_file_name, 'r')):
        word_list.append(line.strip())

    feat_lst = []
    for idx, word in enumerate(word_list):
        if word in word2idx.keys():
            feat_lst.append(feats[word2idx[word]])
        else:
            print('word \'{}\' not found...'.format(word))
            feat_lst.append(np.random.randint(low=-1, high=1, size=config.dim))
    feat_lst = [feat.tolist() for feat in feat_lst]

    check_point = {'feats': feat_lst, 'epoch': cur_epoch, 'log_dir': tb_log_dir}
    json.dump(check_point, open(config.valida_ckpt_dir, 'w'))
