from tqdm import tqdm
import torch

from common import config


class DataProvider():
    def __init__(self, input_file):
        self.input_file = input_file
        text_file = open(input_file, 'r', encoding='utf-8')

        vocab = {}

        def is_word(_word):
            for ch in _word:
                if (ch < '0' or ch > '9') and (ch < 'a' or ch > 'z'):
                    return False
            return True

        for idx, line in tqdm(enumerate(text_file), desc='Reading corpus...'):
            line_words = line.split()
            for word in line_words:
                if len(word) > 1 and is_word(word):
                    if word not in vocab.keys():
                        vocab[word] = 0
                    vocab[word] = vocab[word] + 1

        _vocab = []
        for word, word_cnt in vocab.items():
            if word_cnt >= config.word_min_cnt:
                _vocab.append(word)
        vocab = _vocab

        word2idx = {}
        for idx, word in enumerate(vocab):
            word2idx[word] = idx

        self.word2idx = word2idx
        self.vocab = vocab

    def get_training_set(self, set_size):
        text_file = open(self.input_file, 'r', encoding='utf-8')

        training_set = []

        for idx, line in tqdm(enumerate(text_file), desc='preparing dataset...'):
            line_words = line.split()
            for idx, word in enumerate(line_words):
                beg = max(0, idx - config.window)
                end = min(len(line_words) - 1, idx + config.window)
                for pos_idx in range(beg, end + 1):
                    if pos_idx == idx:
                        continue

                    anchor = line_words[idx]
                    positive = line_words[pos_idx]

                    if (anchor in self.vocab and positive in self.vocab):
                        anchor_idx = self.word2idx[anchor]
                        positive_idx = self.word2idx[positive]
                        training_set.append(((anchor_idx, positive_idx), 1))

        training_set = training_set[:set_size]

        x = torch.Tensor([training_pair[0] for training_pair in training_set])
        y = torch.Tensor([training_pair[1] for training_pair in training_set])
        torch_dataset = torch.utils.data.TensorDataset(x, y)

        return torch_dataset

    def get_voc(self):
        return self.vocab

    def get_voc_size(self):
        return len(self.vocab)
