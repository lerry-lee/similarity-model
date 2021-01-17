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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import numpy as np
import logging
import multiprocessing
import time

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from ernie.classification.reader.task_reader import XWReader
from ernie.classification.model.ernie import ErnieConfig
from ernie.classification.finetune.classifier import create_model

from ernie.classification.utils.args import print_arguments, check_cuda, prepare_logger, ArgumentGroup
from ernie.classification.utils.init import init_pretraining_params
from ernie.classification.utils.args import InferArguments

log = logging.getLogger()


# yapf: enable.

def main(args):
    reader = XWReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=False,
        is_inference=True)

    assert args.save_inference_model_path, "args save_inference_model_path should be set for prediction"
    _, ckpt_dir = os.path.split(args.init_checkpoint.rstrip('/'))
    dir_name = ckpt_dir + '_inference_model'
    model_path = os.path.join(args.save_inference_model_path, dir_name)

    # 如果存在inference_model 则不会再去save一次
    if os.path.isdir(model_path):
        log.info("{} already exist,directly load".format(model_path))
    else:
        ernie_config = ErnieConfig(args.ernie_config_path)
        ernie_config.print_config()
        predict_prog = fluid.Program()
        predict_startup = fluid.Program()
        with fluid.program_guard(predict_prog, predict_startup):
            with fluid.unique_name.guard():
                predict_pyreader, probs, feed_target_names = create_model(
                    args,
                    pyreader_name='predict_reader',
                    ernie_config=ernie_config,
                    is_classify=True,
                    is_prediction=True)

        predict_prog = predict_prog.clone(for_test=True)

        if args.use_cuda:
            place = fluid.CUDAPlace(0)
            dev_count = fluid.core.get_cuda_device_count()
        else:
            place = fluid.CPUPlace()
            dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(predict_startup)

        if args.init_checkpoint:
            init_pretraining_params(exe, args.init_checkpoint, predict_prog)
        else:
            raise ValueError("args 'init_checkpoint' should be set for prediction!")

        log.info("save inference model to %s" % model_path)
        fluid.io.save_inference_model(
            model_path,
            feed_target_names, [probs],
            exe,
            main_program=predict_prog)

    # Set config
    # config = AnalysisConfig(args.model_dir)
    # config = AnalysisConfig(os.path.join(model_path, "__model__"), os.path.join(model_path, ""))
    config = AnalysisConfig(model_path)
    if not args.use_cuda:
        log.info("disable gpu")
        config.disable_gpu()
        config.switch_ir_optim(True)
    else:
        log.info("using gpu")
        config.enable_use_gpu(1024)

    # Create PaddlePredictor
    predictor = create_paddle_predictor(config)

    return reader, predictor


# 持续预测
def run_predict(args):
    predict_data_generator = args.reader.data_generator(
        input_file=args.predict_set,
        batch_size=args.batch_size,
        epoch=1,
        shuffle=False)

    log.info("-------------- prediction results --------------")
    np.set_printoptions(precision=4, suppress=True)
    index = 0
    total_time = 0

    my_list = []
    for sample in predict_data_generator():
        src_ids = sample[0]
        sent_ids = sample[1]
        pos_ids = sample[2]
        task_ids = sample[3]
        input_mask = sample[4]

        inputs = [array2tensor(ndarray) for ndarray in [src_ids, sent_ids, pos_ids, input_mask]]
        begin_time = time.time()
        outputs = args.predictor.run(inputs)
        end_time = time.time()
        total_time += end_time - begin_time

        # parse outputs
        output = outputs[0]
        batch_result = output.as_ndarray()

        for single_example_probs in batch_result:
            # 在这停顿，返回预测为1的概率
            my_list.append(single_example_probs[1])
            # print('\t'.join(map(str, single_example_probs.tolist())))
            index += 1
    try:
        log.info(
            "qps:{}\ttotal_time:{}\ttotal_example:{}\tbatch_size:{}".format(index / total_time, total_time, index,
                                                                            args.batch_size))
    except ZeroDivisionError as e:
        pass

    return my_list


def array2tensor(ndarray):
    """ convert numpy array to PaddleTensor"""
    assert isinstance(ndarray, np.ndarray), "input type must be np.ndarray"
    tensor = PaddleTensor(data=ndarray)
    return tensor


def model_init(args_dict):
    '''
    模型初始化
    Args:
        args_dict: 参数词典：推理时

    Returns:初始化变量：reader、predictor

    '''
    prepare_logger(log)

    args = InferArguments()  # 创建对象
    args.load_args(args_dict)  # 加载参数
    args.display()  # 打印参数
    reader, predictor = main(args)
    args.model_init(reader, predictor)

    return args


def model_predict(args):
    '''
    模型推理
    Args:
        args: 参数

    Returns:推理结果（相似度list）

    '''
    return run_predict(args)
