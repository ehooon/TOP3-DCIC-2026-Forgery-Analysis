# 推理
## 推理硬件需求

- SAM 模型仅需 8GB 显存即可进行 batch_size 为 1 的推理
- Qwen3.5 9B 需要至少 24 GB 显存来保证可以正常输出（量化可以更少，但会降低本任务上的精度）

## 环境配置
1. 首先安装 uv：
```
pip install uv
```
uv 是一种基于 Rust 的包管理工具，比 pip 解析更快，且有更强的处理依赖冲突能力

2. 使用命令安装指定版本的 torch 和 torchvision
```
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url https://download.pytorch.org/whl/cu128
```
3. 安装Qwen3.5推荐版本的VLLM（此处最好使用 uv 进行安装，否则可能出现依赖冲突）
```
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

4. 安装其他库

```
pip install -r requirement.txt
```
5. 安装 unsloth 库
```
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```
6. 强行安装指指定版本的 transformers 库

```
pip install --upgrade transformers==5.3.0
```
如果出现类似如下报错，请无视，项目可正常运行（该报错为VLLM开发版固有报错，不影响运行）
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
compressed-tensors 0.14.0.1 requires transformers<5.0.0, but you have transformers 5.4.0 which is incompatible.
vllm 0.18.1rc1.dev186+gbecaed6ec requires transformers<5,>=4.56.0, but you have transformers 5.4.0 which is incompatible.
```

## 运行方法
命令格式
```
bash scripts/stage2_final/inference.sh --input_path /your/custom/image/dir --output_path /your/custom/output/result.csv
```
也可以不输入指令，移动数据到data目录，直接运行脚本，默认数据集路径存放了运行需要的数据集;
> 为了方便复现，data目录保存了比赛主办方给出的训练集和复赛测试集的子集，其中训练集存放了正负样本各10个，测试集存放了10个样本。

## 环境说明

### 本地运行说明

复赛推理时使用 Pro 6000 BlackWell 96G 单卡，我们保留了一个推理时可以同时运行SAM和Qwen3.5的pip list：piplist-inference.txt

### 微小精度波动说明
> 由于训练时SAM模型使用以 cuda 版本 13.0 为核心的环境，Qwen3.5 微调使用的为 cdua 版本 12.8 为核心的环境，虽然一般情况下，推理不受影响，但我们仍保留了复赛训练时使用的两个的pip list：
> SAM 模型为 piplist-SAM.txt
> Qwen3.5 为 piplist-Qwen.txt

## 推理时间需求
使用 Pro 6000 BlackWell 完成全部推理需要约 20 分钟，如果更换 L40/4090*2，需要约30分钟


## 日志说明
复赛日志文件进行了两次记录，首次记录为 SAM 模型训练时记录，由于赛前未收到需要按step保存log的消息，复赛结束后进行了复现训练，在完全复现SAM的情况下保存了新的日志；

而 Qwen3.5-9B 微调时使用的库无自动截取和保存日志的功能，因此日志为复现训练时输出；

输出日志时使用"add"模式，不删除旧日志，仅追加新日志

## 模型来源说明
SAM 模型为 Meta 开源的模型，Qwen模型为Qwen3.5-9B

## 数据说明
为避免版权和数据泄露问题，本项目未使用任何外部数据

# 训练

## 训练硬件需求
```
$ nvidia-smi
Thu Mar 26 17:25:32 2026       
+-----------------------------------------------------------------------------------------+  
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |  
+-----------------------------------------+------------------------+----------------------+  
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |  
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |  
|                                         |                        |               MIG M. |  
|=========================================+========================+======================|  
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:18:00.0 Off |                  Off |  
| 39%   63C    P1            299W /  300W |   41449MiB /  97887MiB |     93%      Default |  
|                                         |                        |                  N/A |  
+-----------------------------------------+------------------------+----------------------+  
```
## 训练时长
- SAM 模型
    - 训练总计 5 折和 1 个全量数据训练，5 折训练 10 轮，自动折内验证出最优，全量训练 5 轮；
    - 原因：考虑到全量的训练数据是 1000 张，5 折时是800张，5 折平均在 7~8 轮时达到最优，因此判断全量数据训练时，5~6 折达到最优，此后会过拟合
    - 5 折时每轮训练时间约 3 分 30 秒，全量时约 4 分 10 秒，在同一使用 Pro 6000 BlackWell 的情况下，需要约 3.3 小时完成 SAM 模型的训练
    - batch_size 设置为 2 时，需要 42GB 显存用于训练
- Qwen3.5 9B
    - 使用 Unsloth 提供的开源代码和开源库进行微调，完成算子库自动编译后，需要至少 1 小时进行训练，此项微调由于有 Unsloth 优化，仅需 18GB 显存
    - 算子库编译受硬件平台影响，和此次训练相同的环境，需要大概 30 分钟进行算子库自动优化和编译

## 训练复现的要求
推理时已安装训练时需要的库和包，直接运行指令即可
```
bash scripts/stage2_final/train.sh
```

但需要注意，训练完成后仅会按照预设的路径输出模型，不会自动进行推理，需要使用前面的推理方法进行推理。

## 训练完全复现的要求
- SAM 模型需要 2.10.0+cu130 版本的 torch，完整环境参考文件 `piplist-SAM.txt`
- Qwen 模型需要 2.10.0+cu128 版本的 torch，完整环境参考文件 `piplist-Qwen.txt`

直接按照下列流程也可复现训练，但由于自动安装配置环境，可能出现细微的精度波动

### SAM 模型训练复现流程
1. 安装对应的库
```
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install -r requirements-SAM.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

2. 修改路径
代码 `train_sam.py` 的第 24~35 行为路径配置，已写有注释，根据注释配置即可，项目需要的预训练权重也一并打包存放在 weight 目录，可直接使用

3. 运行代码，等待完成即可
```
python train_sam.py
```

### Qwen3.5 9B 微调复现流程
1. 安装对应的库

```
pip install -r requirements-Qwen.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

2. 获取 Qwen3.5 9B 模型
由于 Qwen3.5 9B 模型较大，因此未进行重复打包，运行魔搭平台指令即可直接获取
```
modelscope download --model Qwen/Qwen3.5-9B
```

3. 修改路径 
代码`LoRA.py`的第 27~33 行为配置区域，其中参数 MODEL_NAME 为模型路径，RAW_DATA_ROOT 为训练集路径，CLEAN_DATA_ROOT 为预处理图片存放路径，OUTPUT_DIR 为输出路径

4. 进行微调
```
python LoRA.py
```

## 版权说明
- 数据集为赛事主办方版权所有，因此此处不放置任何数据集，仅保持相同的数据集结构
- base 模型均为开源直接搜索名称即可获得，篡改检测的微调模型基于：[https://github.com/siriusPRX/ForensicsSAM](https://github.com/siriusPRX/ForensicsSAM)；

## 方案说明
1. 使用 SAM + ForensicsSAM 针对比赛数据集微调，同时训练分类和分割任务，加了一些后处理，详见 inference.py
2. 使用 Qwen3.5-9B 进行可解释化，Qwen3.5-9B 仅基于训练集微调
3. 使用 Qwen3.5-9B 对分类阈值在 0.48~0.55 范围内的图像进行分类结果修正

## 联系方式
交流请邮件：1085227472@qq.com

或直接提出 issue