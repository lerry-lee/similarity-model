from flask import Flask, request, jsonify
from gevent import pywsgi
import argparse
import ernie.ernie_handler as ernie

import logging

log = logging.getLogger(__name__)

app = Flask(__name__)


def get_parser():
    """
    命令行参数
    """
    parser = argparse.ArgumentParser(description="相似度模型服务启动参数")
    parser.add_argument("--port", type=int, default=6100, required=False, help="服务对外访问端口")
    parser.add_argument("--model", type=str, default="ernie", required=False, help="所使用的相似度计算模型")

    return parser


@app.route("/hello", methods=["get"])
def test():
    return "hello world"


@app.route("/calculate_similarity", methods=["post"])
def calculate_similarity():
    """
    请求参数：
        text_list1-->list<str>
        text_list2-->list<str>
    Returns:
        {
            "scores":similarities-->list<float>
        }
    """
    text_list1 = request.form.getlist("text_list1")
    text_list2 = request.form.getlist("text_list2")

    similarities = ernie.predict(init_model=init_model, text_list1=text_list1, text_list2=text_list2)

    # 将JSON输出转换为Response具有 application/json mimetype的对象
    return jsonify(scores=similarities)


def ge_model_handler(model_name):
    """
    获得模型的handler：封装模型初始化、预测等方法的py
    Args:
        model_name: 模型名

    Returns: 模型的handler

    """
    if model_name == "ernie":
        return ernie
    # 可以类比扩展其他模型
    elif model_name == "xx":
        return
    else:
        log.error("model\"%s\" not found", model_name)
        return


if __name__ == "__main__":
    # 读命令行参数
    args = get_parser().parse_args()
    port = args.port
    model = args.model

    # 获得模型handler
    model_handler = ge_model_handler(model)
    # 模型初始化
    init_model = model_handler.init()
    log.info("%s模型初始化完成", model)

    # 启动server
    server = pywsgi.WSGIServer(('', port), app)
    server.serve_forever()
