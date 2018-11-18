LOG_DIR = ./logs
DATA_DIR = ./data
DATA_URL = https://ml.qizy.tech/wp-content/uploads/2018/11/quora_questions_gbk_fixed.txt

all:


train:
	./train.py

tb:
	tensorboard --logdir $(LOG_DIR)

eval:
	./tester.py

get data:
	wget -P $(DATA_DIR) $(DATA_URL)