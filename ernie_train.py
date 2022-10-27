# -*- coding: utf-8 -*-
# @PythonName: ernie_train.py
# @Author: lerry_li
# @CreateDate: 2022/10/27
# @Description ernie训练

from ernie.classification.run_classifier import model_train
import time
import paddle
import logging

log = logging.getLogger(__name__)
paddle.enable_static()


def get_args_dict():
    """
        ernie训练参数，不常用的可以使用默认值
    """
    batch_size = 16
    init_pretraining = "data/ernie/pretrained_model"  # 预训练模型参数所在路径
    train_set = "data/train/train_demo.txt"  # 训练集
    dev_set = "data/train/dev_demo.txt"  # 验证集
    epoch = 3
    learning_rate = 2e-5
    model_file_name = "model_demo"  # 训练完成的模型名，将在路径checkpoints_path下
    checkpoints_path = "data/ernie/checkpoints"
    # 创建参数字典
    args_dict = {
        "use_cuda": True,  # 是否使用GPU
        "verbose": True,
        "do_train": True,
        "do_val": True,
        "do_test": False,
        "batch_size": batch_size,
        "init_pretraining_params": "{}/params".format(init_pretraining),
        "train_set": train_set,
        "dev_set": dev_set,
        "vocab_path": "{}/vocab.txt".format(init_pretraining),
        "checkpoints": checkpoints_path,
        "save_steps": 100000,
        "weight_decay": 0.01,
        "warmup_proportion": 0.0,
        "validation_steps": 100,
        "epoch": epoch,
        "max_seq_len": 128,
        "ernie_config_path": "{}/ernie_config.json".format(init_pretraining),
        "learning_rate": learning_rate,
        "skip_steps": 10,
        "num_iteration_per_drop_scope": 1,
        "num_labels": 2,
        "random_seed": 1,
        "task_id": 2,
        "model_file_name": model_file_name
    }

    return args_dict


if __name__ == '__main__':
    t0 = time.time()
    model_train(get_args_dict())
    t1 = time.time()
    log.info("训练耗时：{}s".format(t1 - t0))
