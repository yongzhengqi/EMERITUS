#!/usr/bin/env python3

from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

word_num = 3000

import numpy as np
from common import config
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json
import time


def read_word_list(fin):
    word_list = []
    word2idx = {}
    for idx, line in enumerate(fin):
        word_list.append(line.strip())
        word2idx[line.strip()] = idx
    fin.close()
    return word_list, word2idx


def read_gold_standard(fin, word2idx):
    gold_standard = []
    for line in fin:
        word1, word2, sim = line.strip().split()
        if word1 not in word2idx or word2 not in word2idx:
            continue
        gold_standard.append((word2idx[word1], word2idx[word2], float(sim)))
    fin.close()
    return gold_standard


def eval_ITC(gold_standard, matrix):
    rs = 0
    my_similarity = []
    for wid1, wid2, _ in gold_standard:
        my_similarity.append(matrix[wid1][wid2])
    n = len(my_similarity)

    my_similarity_rank = {item[1]: item[0] for item in
                          enumerate(sorted(range(len(my_similarity)), key=lambda k: my_similarity[k]))}
    gold_similarity_rank = sorted(enumerate(gold_standard), key=lambda x: x[1][2])
    for rkg in range(len(gold_similarity_rank)):
        pair_id = gold_similarity_rank[rkg][0]
        rkm = my_similarity_rank[pair_id]
        rs += (rkg - rkm) ** 2
    rs = 1 - 6 * (rs) / n / (n * n - 1)
    return rs


def get_norm(a):
    return (a ** 2).sum() ** 0.5


def similarity(vec_a, vec_b):
    vec_b, vec_a = np.array(vec_a), np.array(vec_b)
    dot = np.dot(vec_a, vec_b)
    cos_dis = dot / get_norm(vec_a) / get_norm(vec_b)
    return cos_dis


def get_matrix(feat_lst):
    dis_matrix = []
    for i in tqdm(range(word_num), desc='creating distance matrix'):
        dis_matrix.append([])
        for j in range(word_num):
            dis = similarity(feat_lst[i], feat_lst[j])
            dis_matrix[-1].append(dis)

    return dis_matrix


if __name__ == "__main__":
    word_list_file_name = 'evaluation/ITC/word_list.txt'
    gold_standard_file_name = 'evaluation/ITC/wordsim_quora'

    word_list, word2idx = read_word_list(open(word_list_file_name))
    gold_standard = read_gold_standard(open(gold_standard_file_name), word2idx)

    last_epoch = -1

    while last_epoch < config.max_epoch - 1:
        data_pack = json.load(open(config.valida_ckpt_dir, 'r'))
        feats = data_pack['feats']
        epoch = data_pack['epoch']
        log_dir = data_pack['log_dir']

        tb_writer = SummaryWriter(log_dir)

        if epoch == last_epoch:
            print('latest model is the same, sleep for 30s')
            time.sleep(30)
            continue

        avg_norm = np.array([get_norm(np.array(feat)) for feat in feats]).mean()
        print('features\' average norm: {}'.format(avg_norm))
        tb_writer.add_scalar('average norm', avg_norm, epoch)

        last_epoch = epoch

        print('evaluating epoch {}...'.format(epoch))

        matrix = get_matrix(feats)
        validation_var = eval_ITC(gold_standard, matrix)
        tb_writer.add_scalar('validation score', validation_var, epoch)

        print('evaluation done, score = {}'.format(validation_var))

