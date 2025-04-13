# DeepSeek_SFT
DeepSeek_Supervised_FineTune_LoRA

# 1、问题描述

加载DeepSeek官网提供的预训练权重，采用LoRA方法，在医疗数据集上进行SFT微调。

DeepSeek预训练权重：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

医疗数据集：https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

FreedomIntelligence/medical-o1-reasoning-SFT 是一个用于医疗领域的数据集，旨在为医疗大模型的训练和优化提供支持，推动医疗领域的智能化发展。该数据集是使用GPT-4o构建的，GPT-4o会搜索可验证医学问题的解决方案，并通过医学验证器对这些方案进行验证。

该数据集包含24772个问答样本，每个问答样本由"Question"、"Complex_CoT"、"Response"三部分组成，曾用于微调Huatuo GPT-o1，这是一款专为高级医学推理而设计的医疗领域大语言模型。



# 2、环境配置

下载DeepSeek-R1-Distill-Qwen-7B权重文件：

``` bash 
# 下载库
pip install -U huggingface_hub hf_transfer -i https://pypi.tuna.tsinghua.edu.cn/simple
# 配置环境变量
export HF_ENDPOINT=https://hf-mirror.com
# 下载数据集/模型
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir your_path
```

# 3、代码运行

``` bash 
# 模型训练（LoRA SFT微调）
python train.py
# 加载SFT后保存的LoRA权重，模型推理
python test.py
# 可视化损失函数
python plot_loss.py
```

# 4、实验结果展示
训练过程中的损失函数值如图所示：

提出问题：

原始模型权重（DeepSeek-R1-Distill-Qwen-7B）推理结果：

经过SFT微调后（logs/checkpoint-23528/）推理结果：

从实验结果来看，经过SFT微调之后，模型回复的画风有所转变（语言更加直白，表述更加浅显易懂），回答结果的准确度暂时无法评价。


