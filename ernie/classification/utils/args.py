#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Arguments for configuration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import six
import os
import sys
import argparse
import logging

import paddle.fluid as fluid

log = logging.getLogger(__name__)


def prepare_logger(logger, debug=False, save_to_file=None):
    # del logger.handlers[0]
    # 删除paddle创建的handler，否则日志输出2次
    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d]:\t%(message)s')
    console_hdl = logging.StreamHandler()
    console_hdl.setFormatter(formatter)
    logger.addHandler(console_hdl)
    if save_to_file is not None and not os.path.exists(save_to_file):
        file_hdl = logging.FileHandler(save_to_file)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, positional_arg=False, **kwargs):
        prefix = "" if positional_arg else "--"
        type = str2bool if type == bool else type
        self._group.add_argument(
            prefix + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


def print_arguments(args):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        if arg == 'predict_sets': continue
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')


def check_cuda(use_cuda, err= \
        "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
        Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on "
        "CPU.\n"
               ):
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            log.error(err)
            sys.exit(1)
    except Exception as e:
        pass


class InferArguments(object):
    '''
    模型参数：推理时
    '''

    def __init__(self):
        # model arguments 模型主要参数
        self.ernie_config_path = None
        self.init_checkpoint = None
        self.save_inference_model_path = "inference_model"
        self.use_fp16 = False
        self.num_labels = 2
        # data arguments 数据主要参数
        self.predict_set = []
        self.vocab_path = None
        self.label_map_config = None
        self.max_seq_len = 128
        self.batch_size = 32
        self.do_lower_case = True
        # run_type arguments 执行类型参数
        self.use_cuda = True
        self.do_prediction = True
        self.reader = None
        self.predictor = None

    def load_args(self, args_dict):
        '''
        加载模型参数：推理时
        '''
        self.ernie_config_path = args_dict['ernie_config_path']
        self.init_checkpoint = args_dict['init_checkpoint']
        self.vocab_path = args_dict['vocab_path']
        self.batch_size = args_dict['batch_size']
        self.use_cuda = args_dict["use_cuda"]
        self.save_inference_model_path = args_dict["save_inference_model_path"]

    def model_init(self, reader, predictor):
        '''
        模型初始化：变量保存
        '''
        self.reader = reader
        self.predictor = predictor

    def display(self):
        log.info('-----------  Configuration Arguments -----------')
        for name, value in vars(self).items():
            if name == "predict_set" or value is None:
                continue
            log.info("{}:{}".format(name, value))
        log.info('------------------------------------------------')


class TrainArguments(object):
    '''
    模型参数：训练时
    '''

    def __init__(self, dict):
        # model
        self.ernie_config_path = dict["ernie_config_path"]
        self.init_checkpoint = dict["init_checkpoint"] if "init_checkpoint" in dict.keys() else None
        self.init_pretraining_params = dict[
            "init_pretraining_params"] if "init_pretraining_params" in dict.keys() else None
        self.checkpoints = dict["checkpoints"]
        self.is_classify = True
        self.is_regression = False
        self.task_id = dict["task_id"]
        # training
        self.epoch = dict["epoch"]
        self.learning_rate = dict["learning_rate"]
        self.lr_scheduler = "linear_warmup_decay"
        self.weight_decay = dict["weight_decay"]
        self.warmup_proportion = dict["warmup_proportion"]
        self.save_steps = dict["save_steps"]
        self.validation_steps = dict["validation_steps"]
        self.use_fp16 = False
        self.use_dynamic_loss_scaling = True
        self.init_loss_scaling = 102400
        self.test_save = "./checkpoints/test_result"
        self.metric = "simple_accuracy"
        self.incr_every_n_steps = 100
        self.decr_every_n_nan_or_inf = 2
        self.incr_ratio = 2.0
        self.decr_ratio = 0.8
        self.model_file_name = dict["model_file_name"]
        # logging
        self.skip_steps = dict["skip_steps"]
        self.verbose = dict["verbose"]
        # data
        self.tokenizer = "FullTokenizer"
        self.train_set = dict["train_set"]
        self.test_set = None
        self.dev_set = dict["dev_set"]
        self.vocab_path = dict["vocab_path"]
        self.max_seq_len = dict["max_seq_len"]
        self.batch_size = dict["batch_size"]
        self.predict_batch_size = None
        self.in_tokens = False
        self.do_lower_case = True
        self.random_seed = dict["random_seed"]
        self.label_map_config = None
        self.num_labels = dict["num_labels"]
        self.diagnostic = None
        self.diagnostic_save = None
        self.max_query_length = 64
        self.max_answer_length = 100
        self.doc_stride = 128
        self.n_best_size = 20
        self.chunk_scheme = "IOB"
        # run_type
        self.use_cuda = dict["use_cuda"]
        self.is_distributed = False
        self.use_fast_executor = False
        self.num_iteration_per_drop_scope = dict["num_iteration_per_drop_scope"]
        self.do_train = dict["do_train"]
        self.do_val = dict["do_val"]
        self.do_test = dict["do_test"]
        self.use_multi_gpu_test = False
        self.metrics = True
        self.shuffle = True
        self.for_cn = True
        # The flag indicating whether to run the task for continuous evaluation
        self.enable_ce = False

    def display(self):
        log.info('-----------  Configuration Arguments -----------')
        for name, value in vars(self).items():
            if value is not None:
                log.info("{}:{}".format(name, value))
        log.info('------------------------------------------------')
