# -*- coding: utf-8 -*-
# @PythonName: ernie_handler.py
# @Author: lerry_li
# @CreateDate: 2021/1/17
# @Description 封装ernie模型的常用方法

from ernie.classification.infer_classifyer import model_init, model_predict
import json
import numpy as np
import paddle

paddle.enable_static()

model_conf = "conf/model_conf.json"


def init():
    """
    模型初始化

    Returns: 初始化后的模型modeler

    """
    with open(model_conf, "r", encoding="utf-8") as file:
        args = json.load(file)["ernie"]

    return model_init(args)


def predict(init_model, text_list1, text_list2):
    """
    使用模型预测，计算两个文本的相似度
    Args:
        init_model: 初始化的模型
        text_list1: 文本1列表
        text_list2: 文本2列表

    Returns: 相似度列表

    """
    predict_set = []
    for text1, text2 in zip(text_list1, text_list2):
        predict_set.append(text1 + "\t" + text2 + "\t0")

    init_model.predict_set = predict_set

    probs = model_predict(init_model)
    probs_float64 = np.asarray(probs, dtype=float).tolist()

    return probs_float64
