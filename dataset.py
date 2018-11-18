from tqdm import tqdm
import torch
import math
import numpy as np
import json
import torch.utils.data as Data

from common import config


class DataProvider:
    class NegativeTable:
        def __init__(self, vocab, word2idx):
            prob_sum = 0.0
            for word, cnt in vocab.items():
                if cnt >= config.word_min_cnt:
                    prob_sum += math.pow(cnt, config.ng_pow)

            neg_table = []
            for word, cnt in tqdm(vocab.items(), desc='Initializing Negative Table'):
                if cnt >= config.word_min_cnt:
                    ins = math.pow(cnt, config.ng_pow) / prob_sum * config.ng_table_sz
                    id = word2idx[word]
                    for i in range(int(ins)):
                        neg_table.append(id)

            while len(neg_table) < config.ng_table_sz:
                neg_table.append(0)

            self.neg_table = neg_table

        def sample(self, x):
            idxs = np.random.randint(low=0, high=config.ng_table_sz, size=x)
            return [self.neg_table[idx] for idx in idxs]

    def __init__(self, input_file):
        self.input_file = input_file
        text_file = open(input_file, 'r', encoding='utf-8')
        self.input_file_sz = 0

        vocab = {}

        for idx, line in tqdm(enumerate(text_file), desc='Reading corpus'):
            self.input_file_sz += 1
            line_words = line.split()
            for word in line_words:
                if self.is_word(word):
                    if word not in vocab.keys():
                        vocab[word] = 0
                    vocab[word] = vocab[word] + 1

        vocab_lst = []
        for word, word_cnt in vocab.items():
            if word_cnt >= config.word_min_cnt:
                vocab_lst.append(word)
        print('{} words valid'.format(len(vocab_lst)))

        word2idx = {}
        for idx, word in enumerate(vocab_lst):
            word2idx[word] = idx

        for word, word_cnt in vocab.items():
            if word_cnt < config.word_min_cnt:
                word2idx[word] = -1

        self.word2idx = word2idx
        self.vocab = vocab_lst
        self.ntable = self.NegativeTable(vocab, word2idx)

    def get_training_set(self, set_size):
        training_set = []

        if config.saved_training_set is not None:
            print('loading saved training set: {}'.format(config.saved_training_set))
            training_set = json.load(open(config.saved_training_set, 'r', encoding='utf-8'))
            print('using saved training set: {}'.format(config.saved_training_set))
        else:
            text_file = open(self.input_file, 'r', encoding='utf-8')

            for idx, line in tqdm(enumerate(text_file), desc='preparing dataset', total=self.input_file_sz):
                line_words = line.split()
                line_words = [self.word2idx[word] if self.is_word(word) else -1 for word in line_words]
                for idx, word in enumerate(line_words):
                    anchor = line_words[idx]
                    if anchor >= 0:
                        negative_samples = self.ntable.sample(config.ng_k * config.window)
                        for negative_sample in negative_samples:
                            training_set.append([[anchor, negative_sample], 0])

                        beg = max(0, idx - config.window)
                        end = min(len(line_words) - 1, idx + config.window) + 1
                        for pos_idx in range(beg, end):
                            positive = line_words[pos_idx]
                            if pos_idx != idx and positive >= 0:
                                training_set.append([[anchor, positive], 1])

                if set_size is not None and len(training_set) > set_size:
                    break

            # json.dump(training_set, open('./data/training_set.json', 'w', encoding='utf-8'))

        if set_size is not None:
            training_set = training_set[:set_size]

        print('{} pairs ready...'.format(len(training_set)))
        x = torch.LongTensor([pair[0] for pair in training_set])
        y = torch.Tensor([pair[1] for pair in training_set])
        dataset_combined = torch.utils.data.TensorDataset(x, y)

        gpu_num = torch.cuda.device_count()

        dataset_dataloader = Data.DataLoader(
            dataset=dataset_combined,
            batch_size=config.batch_size,  # * gpu_num,
            shuffle=True,
            num_workers=1,
        )

        print('DataLoader ready...')

        return dataset_dataloader

    def get_voc(self):
        return self.vocab

    def get_voc_size(self):
        return len(self.vocab)

    def is_word(self, _word):
        for ch in _word:
            if (ch < '0' or ch > '9') and (ch < 'a' or ch > 'z'):
                return False
        return True
