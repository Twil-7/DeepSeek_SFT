import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset  # 用于从不同的数据源加载各种数据集

# AutoModelForCausalLM, 根据指定的模型名称自动加载适用于因果语言模型任务的模型; AutoTokenizer, 根据指定的模型名称自动加载对应的分词器
# TrainingArguments, 用于配置模型训练过程中的各种参数; logging, 日志模块, 用于控制和管理模型训练、推理过程中的日志输出
# pipeline, 可以简化模型的使用过程，无需手动进行分词、模型推理等操作，直接将输入文本传递给 pipeline 即可得到输出结果
# DataCollatorForLanguageModeling, 用于语言模型训练的数据整理器，它可以将多个样本组合成一个批次，并对输入进行必要的填充和处理
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline, logging, \
    DataCollatorForLanguageModeling
from peft import LoraConfig  # 用于高效参数微调
from trl import SFTTrainer  # 用于监督式微调（Supervised Fine-Tuning，SFT）的训练器

model_name = "./DeepSeek-R1-Distill-Qwen-7B/"
model_save_path = "./save/"

if torch.cuda.is_available():
    print("Using CUDA GPU")
    device = torch.device("cuda")  # ! Use GPU for faster computation
else:
    print("Using CPU")
    device = torch.device("cpu")  # ! Fallback to CPU if GPU is unavailable


def setup_model():

    # Load the tokenizer and ensure proper padding settings.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token    # 将填充标记设置为与句子结束标记相同
    tokenizer.padding_side = "right"    # 将填充标记添加到序列的右侧
    # print(tokenizer.pad_token): <｜end▁of▁sentence｜>

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # 自动将模型映射到可用的设备上
        trust_remote_code=True,    # 允许从远程代码加载模型
        torch_dtype=torch.float16,  # 指定模型使用的数据类型为 float16（半精度浮点数）
        use_cache=False,  # 禁用 KV-cache（设置为 False）可以节省内存，因为缓存会占用额外的内存空间
        max_memory={0: "24GB"},  # 用于设置特定设备的最大内存分配
    )

    # 通过检查模型是否具备特定的优化方法，并在条件满足时调用该方法，实现了内存优化和效率提升
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


def prepare_dataset(tokenizer):

    dataset_path = "./medical_o1_sft_Chinese.json"
    dataset = load_dataset('json', data_files=dataset_path)
    # print(set(dataset)): {'train'}    # json只有一部分，被全划分为"train"
    print(f"Dataset loaded with {len(dataset['train'])} training examples")    # 85

    def format_instruction(sample):
        # 推荐使用合适的prompt模板来得到更好的推理性能，对每个原始样本，格式化训练数据：
        # print(set(sample)): {'question', 'Chain of thought', 'answer'}

        output = f"""请一步一步推理下面的问题：\n\n{sample['Question']}\n\n让我们来一步一步解决：\n\n{sample['Complex_CoT']}\n\n最终的回答：\n\n{sample['Response']}"""
        # print(output)
        return output

    dataset = dataset["train"].train_test_split(train_size=0.95, test_size=0.01, seed=42)
    # print(set(dataset)): {'train', 'test'}

    # Format each training example.
    train_dataset = dataset["train"].map(
        lambda x: {"text": format_instruction(x)},
        remove_columns=dataset["train"].column_names,    # 指定在处理完样本后，移除原始数据集中的所有列，处理完后只包含"text"列
        num_proc=os.cpu_count()  # 设置并行处理的进程数
    )

    # Tokenize the formatted text.
    # Note: max_length is set to 1024 tokens for a balance between speed and context length.
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,    # 如果输入的文本长度超过了分词器的最大允许长度，就会对文本进行截断处理
            padding="max_length",    # 对输入的文本进行填充，使其长度达到分词器的最大允许长度
            return_tensors=None,    # 指定返回的结果不使用张量（tensor）格式，而是返回普通的 Python 列表
        ),
        # remove_columns=["text"],
        num_proc=os.cpu_count()
    )

    """ 说明每段文本都被填充至16384长度 """
    print(f"\nUsing {len(train_dataset)} examples for training")    # 80
    # print(set(train_dataset[0])): {'attention_mask', 'input_ids', 'text'}
    # print(len(train_dataset[0]['attention_mask'])): 16384
    # print(len(train_dataset[0]['input_ids'])): 16384
    # print(len(train_dataset[0]['text'])): 3893

    return train_dataset


def setup_trainer(model, tokenizer, train_dataset, eval_dataset):

    # Configure LoRA for efficient fine-tuning.
    peft_config = LoraConfig(
        lora_alpha=16,  # 用于控制 LoRA 层的自适应强度，它是缩放因子，会影响低秩矩阵对模型参数更新的贡献程度
        lora_dropout=0.1,  # LoRA 层的dropout率
        r=4,  # LoRA 的秩（rank）
        bias="none",    # 设置为 "none" 表示不调整偏置项，即偏置项保持预训练模型中的原始值
        task_type="CAUSAL_LM",    # 表示任务是因果语言模型（Causal Language Modeling）任务
        target_modules=["q_proj", "v_proj"]  # 通过指定这些目标模块，LoRA 只会对这些模块的参数进行低秩调整
    )

    # Set training arguments for the trainer.
    training_args = TrainingArguments(
        output_dir="./logs/",  # 模型检查点和训练输出文件的保存目录
        num_train_epochs=8,  # 训练的轮数
        per_device_train_batch_size=2,  # 每个设备上的训练批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数
        learning_rate=1e-4,  # AdamW优化器的初始学习率
        weight_decay=0.01,  # L2正则化强度，用于防止模型过拟合
        warmup_ratio=0.03,  # 学习率热身阶段的训练步数占总训练步数的百分比
        logging_steps=10,  # 每多少步记录一次训练指标
        save_strategy="epoch",  # 保存检查点的策略，设置为"epoch"表示在每个训练轮次结束时保存检查点
        save_total_limit=8,  # 最多保存的检查点数量
        fp16=True,  # 是否使用 fp16 精度
        bf16=False,  # 是否使用 bfloat16 精度
        optim="adamw_torch_fused",  # 优化器的实现方式，"adam-w_torch_fused" 是使用融合操作的 AdamW 优化器，可提高训练速度
        gradient_checkpointing=True,  # 是否启用梯度检查点，通过重新计算激活值来减少内存使用
        group_by_length=True,  # 是否将长度相似的序列分组，以提高数据处理效率
        max_grad_norm=0.3,  # 梯度裁剪的阈值，用于防止梯度爆炸
        dataloader_num_workers=0,  # 数据加载的子进程数量，设置为0表示使用主进程进行数据加载
        remove_unused_columns=True,  # 是否从数据集中移除未使用的列，以节省内存
        deepspeed=None,  # DeepSpeed优化的配置，这里设置为 None 表示不使用
        local_rank=-1,  # 分布式训练中本地进程的排名，设置为-1表示不进行分布式训练
        ddp_find_unused_parameters=None,  # 在分布式数据并行中处理未使用参数的方式，这里设置为None
        torch_compile=False,  # 是否启用PyTorch 2.0的编译器以加快训练速度
    )

    # 用于处理不同长度的样本，进行掩码操作，并生成注意力掩码
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM does not use masked language modeling.
    )

    # Initialize the SFTTrainer with our model, dataset, and training configurations.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=data_collator,    # 数据整理器，用于处理动态填充
        processing_class=None  # 输入处理类，设置为 None 表示由 SFTTrainer 来处理输入
    )

    return trainer


def main():
    """   Workflow:
          1. Set up the model and tokenizer.
          2. Prepare the training dataset.
          3. Configure and initialize the trainer with LoRA settings.
          4. Run training.
          5. Save the fine-tuned model. """

    print("\nSetting up model...")
    model, tokenizer = setup_model()

    print("\nPreparing dataset...")
    train_dataset = prepare_dataset(tokenizer)

    print("\nSetting up trainer...")
    trainer = setup_trainer(model, tokenizer, train_dataset, None)

    print("\nStarting training...")
    trainer.train()  # ! This is where the fine-tuning happens.

    print("\nSaving model...")
    trainer.model.save_pretrained(model_save_path)  # ! Save the fine-tuned model for later use


if __name__ == "__main__":
    main()
