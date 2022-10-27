# 相似度模型服务

对输入的两个句子进行推理预测，获得它们的语义相似度

# 环境依赖

| 工具 | 说明 |
| --- | --- |
| python | 开发语言 |
| paddlepaddle | 深度学习框架 |
| flask | python web框架 |
| gevent | python 并发框架 |

# 接口说明

## 接口介绍

| 接口名 | 类型 | 说明 |
| --- | --- | --- |
| calculate_similarity | post | 提供句子相似度计算 |

## 参数介绍

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| text_list1 | list<str> | 句子列表1 |
| text_list2 | list<str> | 句子列表2 |

## 返回值介绍

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| probs | list<float> | 相似度列表 |

例如输入是`text_list1=[list1_t1,list1_t2,...]`和`text_list2=[list2_t1,list2_t2,...]`，则输出为`scores=[score1,score2,...]`

其中`score1`是`list1_t1`和`list2_t1`的语义相似度得分，`score2`是`list1_t2`和`list2_t2`的语义相似度得分...

# 启动步骤

## 环境搭建

1. 深度学习环境

对于paddlepadlle深度学习框架，参考[快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html)

或直接使用如下命令安装：

```shell
# gpu版
python -m pip install paddlepaddle-gpu==2.0.2.post100 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
# cpu版
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

2. 模型下载

该项目使用深度语言模型[ERNIE](https://github.com/PaddlePaddle/ERNIE)，并基于预训练模型进行微调

所需的预训练模型是`ERNIE1.0中文Base模型`，点击[下载](https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz)

3. python web环境

对于python web框架flask和并发框架gevent，使用如下命令安装即可：

```shell
pip install flask gevent
```

## 配置参数

`conf/model_conf.json`配置了常见的模型参数，如果这些参数不够满足要求，可以根据实际需求更改，然后重写模型加载前的参数读取方法即可

下面是配置参数的示例：

```json
{
  "ernie": {
    "use_cuda": true,
    "batch_size": 32,
    "init_checkpoint": "data/ernie/pretrained_model/params",
    "ernie_config_path": "data/ernie/pretrained_model/ernie_config.json",
    "vocab_path": "data/ernie/pretrained_model/vocab.txt",
    "save_inference_model_path": "data/ernie/inference_models"
  }
}
```

- `use_cuda`表示是否使用GPU
  
- `init_checkpoint`表示模型要加载的检查点（训练好的模型输出的变量、参数等数据）
  
- `ernie_config_path`和`vocab_path`是预训练模型的参数和词典
  
- `save_inference_model_path`为模型推理时加载的模型文件。

    如果没有训练，可以直接使用预训练模型进行推理预测，即`init_checkpoint`的路径改为预训练模型的`params`即可(如示例)。


## 启动服务

进入该项目的根目录，然后执行命令

```shell
# 不指定参数，使用默认值
python server.py
# 指定参数
python server.py --port 6100 --model ernie
# 可使用`python server.py --help`查看参数说明
```

其中参数说明如下：

| 参数名 | 默认值 | 说明 |
| --- | --- | --- |
| port | 6100 | 开放端口 |
| model | ernie | 指定模型 |


# 另附:训练demo

项目根目录下`ernie_train.py`提供了训练ernie模型的方法，训练集和验证集的格式均为：s1`\t`s2`\t`label
```
北京有多少人	北京的人口数量是多少	1
这个空气净化器有用吗？	空气刘海是怎么样的	0
```

