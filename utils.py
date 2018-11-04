import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common import config


def train(loader, net, rank):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum)

    for step, (batch_x, batch_y) in enumerate(tqdm(loader, position=rank)):
        optimizer.zero_grad()
        output = net(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()


def save_model(words, features, filename):
    output_file = open(filename, 'w', encoding='utf-8')

    for word, feature in zip(words, features):
        output_file.write('{} '.format(word))

        for num in feature:
            output_file.write('{} '.format(num))
        output_file.write('\n')

    output_file.close()
