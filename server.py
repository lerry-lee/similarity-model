from flask import Flask, request
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
    parser.add_argument("port", type=int, default=6100, help="服务对外访问端口")
    parser.add_argument("model", type=str, default="ernie", help="所使用的相似度计算模型")

    return parser


@app.route("/similarity_calculation", methods=["post"])
def similarity_calculation():
    """
    请求参数：
        text1_list-->list<str>
        text2_list-->list<str>
    Returns:
        similarities-->list<float>
    """
    text1_list = request.form.getlist("text1_list")
    text2_list = request.form.getlist("text2_list")

    return init_model.predict(text1_list, text2_list)


def ge_model_handler(model):
    """
    获得模型的handler：封装模型初始化、预测等方法的py
    Args:
        model: 模型名

    Returns: 模型的handler

    """
    if model == "ernie":
        log.info("select ernie as similarity model")
        return ernie
    # 可以类比扩展其他模型
    elif model == "xx":
        return
    else:
        log.error("model\"{}\" not found", model)
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

    # 启动server
    app.run(debug=True, port=port)
