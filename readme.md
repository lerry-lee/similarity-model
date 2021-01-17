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
similarities[prob1,prob2,...]
```
其中，`prob1`为`list1_text1`和`list2_text1`的语义相似度

### 启动步骤
1. 前置条件
- 深度学习模型所需的数据以及所依赖的环境
- flask库

2. 在终端上进入该项目的根目录，然后执行命令
```shell
python server.py -port 6100 -model ernie
```
其中可选参数有两个,`-p`和`-model`，可以使用`python server.py -h`查看参数说明，下面直接介绍它们的含义：
- port  
int，服务对外访问端口，默认为6100
  
- model  
str，所使用的相似度计算模型，默认为"ernie"
   
### 接口说明
- similarity_calculation 
`similarity_calculation`接口用于对外提供相似度计算服务，参数及返回如下：
```text
请求参数：
    text_list1-->list<str>
    text_list2-->list<str>
Returns:
    similarities-->list<float>
```

   
