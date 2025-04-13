import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from datasets import load_dataset    # 用于从不同的数据源加载各种数据集

# AutoModelForCausalLM, 根据指定的模型名称自动加载适用于因果语言模型任务的模型; AutoTokenizer, 根据指定的模型名称自动加载对应的分词器
# TrainingArguments, 用于配置模型训练过程中的各种参数; logging, 日志模块, 用于控制和管理模型训练、推理过程中的日志输出
# pipeline, 可以简化模型的使用过程，无需手动进行分词、模型推理等操作，直接将输入文本传递给 pipeline 即可得到输出结果
# DataCollatorForLanguageModeling, 用于语言模型训练的数据整理器，它可以将多个样本组合成一个批次，并对输入进行必要的填充和处理
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline, logging, DataCollatorForLanguageModeling
from peft import LoraConfig  # 用于高效参数微调
from trl import SFTTrainer  # 用于监督式微调（Supervised Fine-Tuning，SFT）的训练器
from peft import PeftModel, PeftConfig


if torch.cuda.is_available():
    print("Using CUDA GPU")
    device = torch.device("cuda")  # ! Use GPU for faster computation
else:
    print("Using CPU")
    device = torch.device("cpu")  # ! Fallback to CPU if GPU is unavailable


def test_model():

    base_model_name = "./DeepSeek-R1-Distill-Qwen-7B/"
    lora_path = "./logs/checkpoint-23528/"

    # Create an offload directory to manage memory during testing.
    os.makedirs("offload", exist_ok=True)

    # Load the model from the saved fine-tuned checkpoint.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",  # ! Automatically map model to available devices
        trust_remote_code=True,
        torch_dtype=torch.float16,  # ! Use float16 precision for reduced memory usage
        offload_folder="offload",  # Specify offloading directory to handle memory constraints.
        offload_state_dict=True,
        use_cache=False,  # ! Disable KV-cache to save memory during inference
        max_memory={0: "24GB"},  # ! Set maximum memory allocation for device 0
    )

    # 加载tokenizer（从微调路径或基础模型）
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,  # 优先尝试微调路径
        trust_remote_code=True,
        padding_side="right")

    # Setup a text-generation pipeline (with original parameters) for medical reasoning.
    pipe1 = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device_map="auto",  # ! Automatically map model to available devices
        max_new_tokens=32768,  # ! Maximum number of new tokens to generate
        temperature=0.6,  # ! Controls randomness in output (lower value = more deterministic).
        top_p=0.95,  # ! Nucleus sampling to limit the token pool.
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    # Define a test prompt with the recommended "step-by-step" reasoning format.
    test_problem = "请一步一步推理下面的问题：\n\n多饮、多尿、多食伴体重下降的患者，空腹血糖显著升高，可能是什么疾病？"
    # test_problem = "请一步一步推理下面的问题：\n\n消化性溃疡患者，幽门螺杆菌检测阳性，应如何进行根除治疗？需选择哪些药物组合？"
    # test_problem = "请一步一步推理下面的问题：\n\n癫痫患者，应根据发作类型选择哪些抗癫痫药物治疗？需关注哪些药物副作用和疗效评估？"
    # test_problem = "请一步一步推理下面的问题：\n\n对于甲状腺功能减退症患者，应如何进行孕期教育和监测？"
    print("\nTest Problem:", test_problem)

    result1 = pipe1(test_problem, max_new_tokens=32768, temperature=0.6, top_p=0.95, repetition_penalty=1.15)
    print("\nRaw Model Response:", result1[0]["generated_text"])
    
    ##################################################################################################
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()  # 合并权重（若需保留独立适配器则跳过此行）

    # Setup a text-generation pipeline (with parameters LoRA tuned) for medical reasoning.
    pipe2 = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # ! Automatically map model to available devices
        max_new_tokens=32768,  # ! Maximum number of new tokens to generate
        temperature=0.6,  # ! Controls randomness in output (lower value = more deterministic).
        top_p=0.95,  # ! Nucleus sampling to limit the token pool.
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    result2 = pipe2(test_problem, max_new_tokens=32768, temperature=0.6, top_p=0.95, repetition_penalty=1.15)
    print("\nLoRA Model Response:", result2[0]["generated_text"])


def main():

    print("\nTesting model...")
    test_model()


if __name__ == "__main__":
    main()
