#!/usr/bin/env python3

import torch.utils.data as Data
import torch.multiprocessing as mp
from dataset import DataProvider
from utils import *
from model import Net

if __name__ == '__main__':
    print("basic settings:\ninput file name: {}\nwindow size: {}\n dimensionality: {}".format(config.input_filename,
                                                                                              config.window,
                                                                                              config.dim))

    data_provider = DataProvider(config.input_filename)

    net = Net(data_provider.get_voc_size(), config.dim)
    net.share_memory()

    torch_dataset = data_provider.get_training_set(config.dataset_size)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=len(torch_dataset) // config.num_worker,
        shuffle=True,
        num_workers=1,
    )
    loader_itr = iter(loader)

    processes = []

    for rank in range(config.num_worker):
        x, y = next(loader_itr)
        dataset_worker = Data.TensorDataset(x, y)
        loader_worker = Data.DataLoader(
            dataset=dataset_worker,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=1,
        )

        p = mp.Process(target=train, args=(loader_worker, net, rank))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    del loader_itr

    save_model(data_provider.get_voc(), net.fe_0.data.numpy(), config.output_filename)
