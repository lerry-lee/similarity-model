# -*- coding: utf-8 -*-
# @PythonName: ernie_handler.py
# @Author: lerry_li
# @CreateDate: 2021/1/17
# @Description 封装ernie模型的常用方法

from ernie.classification.infer_classifyer import model_init, model_predict
import json

model_conf = "conf/model_conf.json"


def init():
    """
    模型初始化

    Returns: 初始化后的模型modeler

    """
    with open(model_conf, "r", encoding="utf-8") as file:
        args = json.load(file)["ernie"]

    return model_init(args)


def predict(text1_list, text2_list):
    """
    使用模型预测，计算两个文本的相似度
    Args:
        text1_list: 文本1列表
        text2_list: 文本2列表

    Returns: 相似度列表

    """
    predict_set = []
    for text1, text2 in zip(text1_list, text2_list):
        predict_set.append(text1 + "\t" + text2 + "\t0")

    return model_predict(predict_set)
