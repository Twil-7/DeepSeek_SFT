# DeepSeek_SFT
DeepSeek_SFT_LoRA

# 1、问题描述

加载DeepSeek官网提供的预训练权重，采用LoRA方法，在医疗数据集上进行SFT微调。

DeepSeek预训练权重：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

医疗数据集：https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

FreedomIntelligence/medical-o1-reasoning-SFT 是一个用于医疗领域的数据集，旨在为医疗大模型的训练和优化提供支持，推动医疗领域的智能化发展。该数据集通过GPT-4o构建（GPT-4o会搜索可验证医学问题的解决方案，并通过医学验证器对这些方案进行验证），曾用于微调Huatuo GPT-o1，这是一款专为高级医学推理而设计的医疗领域大语言模型。该数据集包含24772个问答样本，每个问答样本由"Question"，"Complex_CoT"，"Response"三部分组成：

<div align="center">
  <img src="./files/data.png" alt="sft_data" width="1200"/>
</div>

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
安装项目依赖
``` bash
pip install -r requirements.txt
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

<div align="center">
  <img src="./files/deepseek_sft.png" alt="sft_loss" width="1200"/>
</div>

给定一个Prompt：
``` bash 
Test Problem: 请一步一步推理下面的问题：

多饮、多尿、多食伴体重下降的患者，空腹血糖显著升高，可能是什么疾病？
```

原始权重（DeepSeek-R1-Distill-Qwen-7B）模型Response如下：
``` bash 
Raw Model Response: 请一步一步推理下面的问题：

多饮、多尿、多食伴体重下降的患者，空腹血糖显著升高，可能是什么疾病？（选项：糖尿病、尿毒症、尿崩症）

<think>
嗯，我现在遇到了一个医学问题需要解决。题目是说，有位患者的症状包括多饮、多尿、多食，同时体重也在下降，而且空腹血糖明显升高。然后要从三个选项中选择可能的疾病：糖尿病、尿毒症、尿崩症。

首先，我得先回忆一下这三种疾病各自的特点和症状，再逐一分析看看哪个符合描述的情况。

先看糖尿病。糖尿病主要分为1型和2型两种。1型糖尿病通常是由于免疫系统攻击胰岛β细胞导致胰岛素缺乏或分泌不足，患者通常会有多饮、多尿的症状，但空腹血糖升高的情况更多出现在2型糖尿病中，因为这种类型是由肥胖或其他代谢异常引起的。此外，糖尿病患者可 能会出现口渴、多尿，体重减轻，特别是在2型糖尿病的情况下，因为糖分摄入过多会导致储存为脂肪，所以体重会下降。不过空腹血糖 升高是否常见呢？

接下来是尿毒症，也就是肾脏功能逐渐受损，无法排出毒素的结果。这种情况会引起多尿，尤其是夜间多尿，因为肾脏负担加重。患者还会出现乏力、食欲不振等全身症状。如果肾功能严重恶化，可能出现高钾血症或者其他电解质紊乱。然而，尿毒症通常伴有严重的并发症，比如感染或氮质血症，而空腹血糖升高似乎不太直接相关，除非合并了糖尿病或者酮症酸中毒等情况。

第三个选项是尿崩症，也称为水 intoxication，是指体内水分过少，导致尿量突然增加的现象。这种情况常见于脱水，可能是由于喝水 太少、饮水不规律、运动过度或其他因素导致体液失衡。尿崩症患者在空腹时容易出现大量排尿，但并不一定会伴随多饮、多食和体重下降。更重要的是，尿崩症本身并不会引起血糖水平的变化，更别提空腹血糖显著升高了。

现在回到原题中的症状：多饮、多尿、多食，体重下降，空腹血糖升高。这些症状看起来像是典型的2型糖尿病的表现。多饮是因为身体 缺水，加上对葡萄糖的需求增加；多尿则与葡萄糖浓度过高有关，因为肾小管重吸收水分的能力降低；多食也是因为身体需要摄取更多的食物来维持能量需求，而体重下降则是由于糖分转化为脂肪储存起来。

尿毒症虽然也会有多尿，但其症状通常更为复杂，并且与血糖变化关系不大。尿崩症主要是由水分不足引发的，不涉及血糖控制的问题。因此综合来看，最可能的答案应该是糖尿病。
</think>

根据提供的症状，最可能的疾病是糖尿病。具体分析如下：

- **多饮**：糖尿病患者因体内葡萄糖浓度升高，对水分需求增加，从而多喝液体以补充丢失的水分。
- **多尿**：由于肾小管对水分的重吸收能力减弱，导致尿量增多。
- **多食**：为了维持能量需求，糖尿病患者需摄入较多的食物。
- **体重下降**：多食的同时可能导致多余的热量转化为脂肪储存，进而导致体重下降。
- **空腹血糖显著升高**：这是2型糖尿病的重要特征，通常与肥胖或其他代谢异常有关。

相比之下，尿毒症和尿崩症并不符合上述所有症状。尿毒症常伴有严重并发症，而尿崩症主要由水分不足引起，与血糖变化无关。因此，综合考虑，糖尿病是最符合条件的诊断。
```

经过SFT微调（logs/checkpoint-23528）模型Response如下：
``` bash 
LoRA Model Response: 请一步一步推理下面的问题：

多饮、多尿、多食伴体重下降的患者，空腹血糖显著升高，可能是什么疾病？请详细说明。

让我们来一步一步解决：

这个病人看起来很严重啊。他总是喝水多，喝完后又尿很多，吃得多但没变胖反而变瘦了。这让我想到了肾功能不全，因为肾脏问题通常会导致这样的症状。

再看看他的血糖情况吧。空腹血糖突然升得很高，这不太寻常呢。正常情况下，这个水平应该是正常的，所以这似乎和糖尿病没什么关系。

不过，我不能忽略其他可能性哦。嗯，首先，我们得考虑下是不是糖尿病肾病。这种病可真不是 joke，它能导致肾功能异常，并且会让 人变得特别渴，尿多，还有尿蛋白什么的。空腹血糖高的话，是不是也有可能是这样引起的？

然后，我想到了尿毒症。这个病也会让尿量增加，而且身体反应可能会有代谢性酸中毒，那空腹血糖高的症状就有点相似了。对，尿毒症确实是个需要考虑的选项。

哦，等等，还有一种可能是急性肾衰竭。这个病同样会引起水分摄入过多，尿量大，而且空腹血糖升高。但是，我需要排除一下，毕竟急性肾衰竭一般会有明显高渗血尿，而不是单纯的尿液增加。

也许在急性期的时候，肾小管重吸收减少，出现高渗透压的情况，而如果肾功能好的话，可能不会马上出现高血糖。

想到这里，我觉得可能还是肾小球肾炎或肾小板综合征更合适一点。这两者都是由免疫系统反应导致的，通常会引发多尿和高血糖，特别是在急性阶段。尤其是当免疫系统被激活时，这些症状会迅速显现出来。

总之，结合所有这些信息来看，最有可能的是肾小球肾炎或者肾小板综合征，特别是考虑到这些症状出现在急性期。

最终的回答：

根据您提供的症状：多饮、多尿、多食伴随体重下降，以及空腹血糖显著升高，我们可以推测患者的病情可能与肾功能异常有关。以下是对几种可能原因的分析：

1. **肾小球肾炎**：
   - 这种慢性肾炎通常是由病毒感染（如CMV）或其他病毒引起，会导致肾小球滤过率降低。
   - 在急性肾小球肾炎发作期间，会出现多尿、多饮以及高血糖的症状，尤其是在急性期。
   - 病情可能在急性期发展较快，因此空腹血糖升高是一个合理的表现。

2. **肾小板综合征**：
   - 该综合征由自身抗体攻击肾小板引起，可能导致肾小球过滤障碍。
   - 同样可能出现多尿和多食，以及体重减轻等症状，尤其是肾小板综合征在急性阶段容易出现。
   - 高血糖也可能在此阶段表现出来，因胰岛素抵抗等代谢变化。

3. **急性肾衰竭**：
   - 虽然急性肾衰竭可以导致多尿和高血糖，但在急性期通常伴有高渗透压的尿液，即血浆渗透压升高。
   - 如果肾功能尚好，则可能没有立即出现高血糖的情况。

4. **糖尿病肾病**：
   - 尿路感染或其他因素可能导致肾小球病变，但空腹血糖显著升高并不常见。
   - 虽然糖尿病肾病可以导致多尿，但空腹血糖显著升高通常提示糖尿病本身存在，而非直接相关于肾小球病变。

综上所述，基于提供的症状和急性期的特征，肾小球肾炎或肾小板综合征的可能性较大。然而，为了准确诊断，建议进行进一步的血液检查、尿常规检测、肾活检等，以确认具体病因。同时，及时的治疗包括控制感染、抗病毒治疗、控制血糖水平等，均有助于改善患者的整体状况。若发现任何异常，请及早就医。希望以上分析对您的理解有所帮助。如有更多疑问，请随时咨询专业医疗人员。

```

从实验结果可以看出，经过SFT微调后，模型回复的画风有所转变（语言更加直白，表述更加浅显易懂），说明2w-3w高质量数据集SFT是有效果的。但回答结果的准确度是否有所提升，目前尚无法评价。


