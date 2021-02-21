# 相似度模型服务
### 功能说明
对两个文本列表进行相似度计算
- 输入
```text
text_list1:[list1_text1,list1_text1,...]
text_list2:[list2_text1,list2_text1,...]
```
- 输出
```text
scores[prob1,prob2,...]
```
其中，`prob1`为`list1_text1`和`list2_text1`的语义相似度得分

### 启动步骤
1. 前置条件
- 深度学习模型所需的数据以及所依赖的环境
- flask,gevent

2. 在终端上进入该项目的根目录，然后执行命令
```shell
python server.py --port 6100 --model ernie
```
其中可选参数有两个,`--port`和`--model`，它们的含义如下：
- port  
int，指定服务对外访问端口，默认为6100
  
- model  
str，指定所使用的相似度计算模型，默认为"ernie"
  
也可以使用`python server.py --help`查看参数说明
   
### 接口说明
- calculate_similarity  
`calculate_similarity`接口用于对外提供相似度计算服务，参数及返回如下：
```text
method:
    post
请求参数：
    text_list1-->list<str>
    text_list2-->list<str>
Returns:
    {
        "scores":similarities-->list<float>
    }
```
可以发起一个post请求`{ip}:{port}/calculate_similarity`测试

### 参数配置

`conf/model_conf.json`配置了常见的模型参数，可以根据实际需求更改，然后重写模型加载前的参数读取方法即可。  
下面给出`ernie`配置参数的示例，`use_cuda`表示是否使用GPU，`init_checkpoint`表示模型要加载的检查点（训练好的模型输出的变量、参数等数据），`ernie_config_path`和`vocab_path`是预训练模型的参数和词典，`save_inference_model_path`为模型推理时加载的模型文件。  
如果没有训练，可以直接使用预训练模型进行推理预测，即`init_checkpoint`的路径改为预训练模型的`params`即可(如示例)。

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

### 数据下载

- ERNIE 1.0 中文Base模型  
包含预训练模型参数、词典 vocab.txt、模型配置 ernie_config.json，[下载地址](https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz)

- ERNIE官方源码地址  
[Gitee](https://gitee.com/paddlepaddle/ERNIE/tree/repro/)
[GitHub](https://github.com/PaddlePaddle/ERNIE)