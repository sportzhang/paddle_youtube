# paddle_youtube
使用百度Paddle框架进行视频分类算法NeXtVLAD视频分类模型。

### 百度飞浆视频分类算法

#### 数据下载

使用Youtube-8M官方链接分别下载[训练集](http://us.data.yt8m.org/2/frame/train/index.html)和[验证集](http://us.data.yt8m.org/2/frame/validate/index.html)。每个链接里各提供了3844个文件的下载地址

```
链接: https://pan.baidu.com/s/1-t5Hb2bpUOdQmLFp9Kg1kw 提取码: di43
```

下载的数据为Frame-level features，下载语句，在Linux系统下运行：

```
curl data.yt8m.org/download.py | partition=2/frame/train mirror=us python
curl data.yt8m.org/download.py | partition=2/frame/validate mirror=us python
curl data.yt8m.org/download.py | partition=2/frame/test mirror=us python
```

总的数据量将近1.5T，所以最好保证有充足的磁盘空间，当然可以下载部分训练数据，指定百分比，比如1%：
```
curl data.yt8m.org/download.py | shard=1,100 partition=2/frame/train mirror=us python
```

 。数据下载完成后，将会得到3844个训练数据文件和3844个验证数据文件（TFRecord格式）。 假设存放视频模型代码库PaddleVideo的主目录为: youtube，进入data/dataset/youtube8m目录


 ```
 cd data/dataset/youtube8m
 ```

 在youtube8m下新建目录tf/train和tf/val


 ```
 mkdir tf && cd tf
 mkdir train && mkdir val
 ```

 并分别将下载的train和validate数据存放在其中。

 #### 数据格式化

 为了适用于PaddlePaddle训练，需要离线将下载好的TFRecord文件格式转成了pickle格式，转换脚本请使用dataset/youtube8m/tf2pkl.py。

 在data/dataset/youtube8m 目录下新建目录pkl/train和pkl/val


 ```
 cd data/dataset/youtube8m

 mkdir pkl && cd pkl

 mkdir train && mkdir val
 ```

 转化文件格式(TFRecord -> pkl)，进入data/dataset/youtube8m目录，运行脚本


 ```
 python tf2pkl.py ./tf/train ./pkl/train
 ```

 和


 ```
 python tf2pkl.py ./tf/val ./pkl/val
 ```

 分别将train和validate数据集转化为pkl文件。tf2pkl.py文件运行时需要两个参数，分别是数据源tf文件存放路径和转化后的pkl文件存放路径。

 - 备注：由于TFRecord文件的读取需要用到Tensorflow，用户要先安装Tensorflow，或者在安装有Tensorflow的环境中转化完数据，再拷贝到data/dataset/youtube8m/pkl目录下。为了避免和PaddlePaddle环境冲突，建议先在其他地方转化完成再将数据拷贝过来。

 #### 生成文件数据路径

 进入data/dataset/youtube8m目录

 ```
 cd youtube/data/dataset/youtube8m
 ls /home/sdb1/youtube/data/dataset/youtube8m/pkl/train/* > train.list
 ls /home/sdb1/youtube/data/dataset/youtube8m/pkl/val/* > val.list
 ls /home/sdb1/youtube/data/dataset/youtube8m/pkl/val/* > test.list
 ls /home/sdb1/youtube/data/dataset/youtube8m/pkl/val/* > infer.list
 ```

 在data/dataset/youtube8m目录下将生成四个文件，train.list，val.list，test.list和infer.list，每一行分别保存了一个pkl文件的绝对路径，示例如下：

 ```
 /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/train/train0471.pkl
 /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/train/train0472.pkl
 /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/train/train0473.pkl
 ...
 ```

 或者

 ```
 /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/val/validate3666.pkl
 /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/val/validate3666.pkl
 /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/youtube8m/pkl/val/validate3666.pkl
 ...
 ```

 - 备注：由于Youtube-8M数据集中test部分的数据没有标签，所以此处使用validate数据做模型评估。

 #### 通过train.py快速进行训练

 ```
 export CUDA_VISIBLE_DEVICES=0
 python train.py --model_name=NEXTVLAD \
		--config=./configs/nextvlad.yaml \
		--log_interval=10 \
		--valid_interval=1 \
		--use_gpu=True \
		--save_dir=./data/checkpoints \
		 --fix_random_seed=False
		
		bash run.sh train NEXTVLAD ./configs/nextvlad.yaml
 ```

 nextvlad.yaml文件中指定了模型以及训练相关的参数：

```
epoch: 6  # 迭代次数
learning_rate: 0.0002  # 学习率
lr_boundary_examples: 2000000
max_iter: 700000
learning_rate_decay: 0.8
l2_penalty: 1e-5
gradient_clip_norm: 1.0
use_gpu: True  # 是否使用GPU
num_gpus: 4  # GPU个数
batch_size: 160																 ```
- 备注，在训练NeXtVLAD模型时使用的是4卡，请修改run.sh中的CUDA_VISIBLE_DEVICES=0,1,2,3
#### 使用预训练模型做finetune
将提供的预训练模型[model](https://paddlemodels.bj.bcebos.com/video_classification/nextvlad_youtube8m.tar.gz)下载到本地，并在上述脚本文件中添加--resume为所保存的模型参数存放路径。

```
python train.py --model_name=NEXTVLAD \
--config=./configs/nextvlad.yaml \
--pretrain=./pretrained_model/nextvlad_youtube8m/NEXTVLAD_epoch5/ \
--log_interval=10 \
--valid_interval=1 \
--use_gpu=True \
--save_dir=./data/checkpoints \
--fix_random_seed=False
```

使用4卡Nvidia Tesla P40，总的batch size数是160。

#### 训练策略

使用Adam优化器，初始learning_rate=0.0002
每2,000,000个样本做一次学习率衰减，learning_rate_decay = 0.8
正则化使用l2_weight_decay = 1e-5

#### 模型测试

可通过如下两种方式进行模型评估:

```
python eval.py --model_name=NEXTVLAD \
--config=./configs/nextvlad.yaml \
--log_interval=1 \
--weights=./data/model/NEXTVLAD_final.pdparams \
--use_gpu=True

bash run.sh eval NEXTVLAD ./configs/nextvlad.yaml
```
	
使用run.sh进行评估时，需要修改脚本中的weights参数指定需要评估的权重。

若未指定--weights参数，脚本会下载已发布模型model进行评估

评估结果以log的形式直接打印输出GAP、Hit@1等精度指标

使用CPU进行评估时，请将use_gpu设置为False

由于youtube-8m提供的数据中test数据集是没有ground truth标签的，所以这里使用validation数据集来做测试。


#### 模型推断

可通过如下两种方式启动模型推断：

	
```
python predict.py --model_name=NEXTVLAD \
--config=configs/nextvlad.yaml \
--log_interval=1 \
--weights=./data/model/NEXTVLAD_final.pdparams \
--filelist=./data/dataset/youtube8m/infer.list \
--use_gpu=True

bash run.sh predict NEXTVLAD ./configs/nextvlad.yaml
```

使用python命令行启动程序时，--filelist参数指定待推断的文件列表，如果不设置，默认为data/dataset/youtube8m/infer.list。--weights参数为训练好的权重参数，如果不设置，程序会自动下载已训练好的权重。这两个参数如果不设置，请不要写在命令行，将会自动使用默 认值。

使用run.sh进行评估时，请修改脚本中的weights参数指定需要用到的权重。

若未指定--weights参数，脚本会下载已发布模型model进行推断

模型推断结果以log的形式直接打印输出，可以看到每个测试样本的分类预测概率。

使用CPU进行预测时，请将use_gpu设置为False

百度Paddlevideo参考链接：
https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo/models/nextvlad#%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83















