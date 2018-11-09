#!/usr/bin/env python3

import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from time import localtime, strftime

from dataset import DataProvider
from utils import *
from model import Net

if __name__ == '__main__':
    print("basic settings:\ninput file name: {}\nwindow size: {}\ndimensionality: {}".format(config.input_filename,
                                                                                             config.window,
                                                                                             config.dim))

    # initialize tensorboard
    tb_log_dir = 'logs/' + strftime("%Y-%m-%d-%H:%M:%S", localtime())
    tb_writer = SummaryWriter(tb_log_dir)

    # initialize dataset
    data_provider = DataProvider(config.input_filename)
    data_loader = data_provider.get_training_set(config.dataset_size)
    loader_itr = iter(data_loader)

    # initialize model
    net = Net(data_provider.get_voc_size(), config.dim)
    net = net.cuda()
    net_multi_gpu = nn.DataParallel(net)
    gpu_num = torch.cuda.device_count()

    # specifying optimizing method
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net_multi_gpu.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.lr_adj_pat)

    # training
    for cur_epoch in tqdm(range(config.max_epoch), desc='training on {} GPUs...'.format(gpu_num)):
        try:
            mini_batch = next(loader_itr)
        except StopIteration:
            loader_itr = iter(data_loader)
            mini_batch = next(loader_itr)

        batched_x, batched_y = mini_batch
        batched_x, batched_y = batched_x.cuda(), batched_y.cuda()
        optimizer.zero_grad()
        output = net_multi_gpu(batched_x)
        loss = criterion(output, batched_y)
        loss.backward()
        optimizer.step()

        if (cur_epoch % config.tb_upd_gap) == 0:
            loss_var = loss.data.cpu().numpy()
            print('training loss: {}'.format(loss_var))
            tb_writer.add_scalar('training loss', loss_var, cur_epoch)
        if (cur_epoch % config.ckpt_save_gap) == 0:
            print('saving check point...')
            embed_vec = net_multi_gpu.module.fe.weight.detach().cpu().numpy()
            save_model(data_provider.get_voc(), embed_vec, './results/{}-epoch.ckpt'.format(cur_epoch))
        if (cur_epoch % config.latest_upd_gap) == 0:
            print('updating latest model...')
            embed_vec = net_multi_gpu.module.fe.weight.detach().cpu().numpy()
            save_features(embed_vec, cur_epoch, tb_log_dir, data_provider.word2idx)
            save_model(data_provider.get_voc(), embed_vec, config.latest_ckpt_dir)
        cur_epoch += 1
        scheduler.step(loss)

    embed_vec = net_multi_gpu.module.fe.weight.detach().cpu().numpy()
    save_model(data_provider.get_voc(), embed_vec, config.output_filename)
