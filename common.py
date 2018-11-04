class Config:
    # the size of windows of skip-gram
    window = 3

    # the number of dimensions of features
    dim = 50

    # where you saved your corpus
    input_filename = 'input'

    # where you want to save the representation of your words
    output_filename = './results/output.{}.window={}.dim={}'.format(input_filename, window, dim)

    # if a word appears less than word_min_cnt times, it will be replaced
    word_min_cnt = 3

    # the learning rate of SGD
    learning_rate = 0.01

    # the momentum for SGD
    momentum = 0.5

    # the max number of sentence used for training
    dataset_size = 500000

    # batch size of SGD
    batch_size = 50

    # how many processes you want to create for training process
    num_worker = 8

config = Config()
