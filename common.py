class Config:
    # the size of windows of skip-gram
    window = 3

    # the number of dimensions of features
    dim = 300

    # where you saved your corpus
    input_filename = './data/quora_questions_gbk_fixed.txt'

    # where you want to save the representation of your words
    output_filename = './results/output.window={}.dim={}'.format(window, dim)

    # if a word appears less than word_min_cnt times, it will be replaced
    word_min_cnt = 30

    # the max number of sentence used for training
    # set to None if you want to ignore this limit
    dataset_size = None

    # batch size of SGD
    batch_size = 2048

    # parameter in Negative sampling
    # see more at https://arxiv.org/abs/1301.3781
    ng_pow = 0.75

    # parameter in Negative sampling
    # see more at https://arxiv.org/abs/1301.3781
    ng_table_sz = 100000000

    # parameter in Negative sampling
    # see more at https://arxiv.org/abs/1301.3781
    ng_k = 5

    # if to lazy load the training set
    saved_training_set = None  # 'data/training_set.json'

    # run how many mini-batches between two updates on tensorboard
    tb_upd_gap = 500

    # run how many mini-batches between updates on saved models
    latest_upd_gap = 5000

    # the gap between check points
    ckpt_save_gap = 5000

    # max mini-batch to train
    max_epoch = 300000

    # where to save check latest models
    latest_ckpt_dir = './results/latest'

    # where to save file for testing on validation set
    valida_ckpt_dir = './results/latest.json'

    # hyper-parameter on optimizing
    # see more at https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    lr_adj_pat = 1e4

    # min learning rate
    # see more at https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    lr_min = 1e-5


config = Config()
