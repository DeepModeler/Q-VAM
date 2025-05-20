import os
import pathlib
import argparse

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, CLIPImageProcessor, PhiConfig, StableLmConfig
from llava_trainer import LLaVATrainer
from models.class_arch import VQAModelphi, VQAModelstablelm
from dataset.vqadataset import make_supervised_data_module
from utils.mm_utils import find_all_linear_names
from utils.utils import TrainingArguments1, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer

import warnings
warnings.filterwarnings("ignore")


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='slake', choices=('pvqa', 'rad', 'slake'))
    parser.add_argument("--model_type", type=str, default="phi", choices=("phi", "stablelm"))
    parser.add_argument("--setting", type=str, default="all", choices=("lora", 'all'))
    parser.add_argument("--bs", type=int, default=16) # train batch size
    parser.add_argument("--e_bs", type=int, default=4) # eval batch size
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.3)

    parser.add_argument("--save_dir", default="./Q-VAM/save_model")
    parser.add_argument("--data_path", default="./Q-VAM/VQA_datasets/train_all.json")
    parser.add_argument("--image_folder", default="./Q-VAM/VQA_datasets/images")
    parser.add_argument("--model_name_or_path", type=str, default="./Q-VAM/checkpoint")
    parser.add_argument("--image_tower_path", type=str, default="./Q-VAM/checkpoint/openai_clip-vit-large-patch14-336")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--mm_vision_select_feature", type=str, default="patch") # cls_patch patch
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--tokenizer_padding_side", type=str, default="right") #right left
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="./Q-VAM/checkpoint/base")
    parser.add_argument("--preK", type=int, default=4)
    parser.add_argument("--ablation", type=str, default="all", choices=("all", "w_SA", "w_CA"))
    parser.add_argument("--plot", type=str, default="no", choices=("yes", "no"))
    parser.add_argument("--zero_path", type=str, default="./Q-VAM/zero2.json")

    # lora configuration
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32) #
    parser.add_argument("--only_lora_ffn", type=str, default='no') #yes no

    args = parser.parse_args()
    return args

# ------------------------------------------------------------------------------------
def train_loop(args, data_module, model, tokenizer):
    training_args = TrainingArguments1(
        output_dir=args.save_dir,
        deepspeed=args.zero_path,
        learning_rate=args.lr,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        tf32=True,
        local_rank=-1,
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0,
        max_grad_norm=0.5,
        save_steps=100,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=args.e_bs,
        per_device_train_batch_size=args.bs,
        num_train_epochs=args.epoch,
        model_max_length=args.model_max_length,
        bf16=True,
        save_strategy="steps",
        save_total_limit=1,
        metric_for_best_model="rougeL",
        report_to="tensorboard",
    )

    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cpu")
    print(f"[INFO] Using device: {device}")
    model.to(device)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module,
                           )

    print('trainer.model.device: ', trainer.model.device)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # model.config.use_cache = True
    save_path = os.path.join(training_args.output_dir, args.model_type)
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(save_path)
            model.save_pretrained(save_path, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(save_path, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=save_path)


# =================================================================================
if __name__ == "__main__":
    args = parse_argument()
    args.model_name_or_path = os.path.join(args.model_name_or_path, args.model_type)
    args.save_dir = os.path.join(args.save_dir, args.model_type + '_preK=' + str(args.preK)+'_abl'+args.ablation)
    args.pretrain_mm_mlp_adapter = os.path.join(args.pretrain_mm_mlp_adapter, args.model_type, 'mm_projector.bin')
    # ------------------------------------------------------------------------------------
    if 'phi' in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  model_max_length=args.model_max_length,
                                                  padding_side="right",
                                                  use_fast=False, )
        tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'}) 
    elif 'stablelm' in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  model_max_length=args.model_max_length,
                                                  padding_side="right",
                                                  use_fast=False, )
        tokenizer.unk_token = '<|reg0|>'
    # ------------------------------------------------------------------------------------
    image_processor = CLIPImageProcessor.from_pretrained(args.image_tower_path)
    if args.model_type=='stablelm':
        config = StableLmConfig.from_pretrained(args.model_name_or_path)
    elif args.model_type=='phi':
        config = PhiConfig.from_pretrained(args.model_name_or_path)
    config.attn_implementation = "flash_attention_2"
    args.tokenizer_model_max_length = tokenizer.model_max_length

    print(args)

    if args.model_type == 'phi':
        model = VQAModelphi.from_pretrained(args.model_name_or_path, args=args, attn_implementation="flash_attention_2")
    elif args.model_type == 'stablelm':
        model = VQAModelstablelm.from_pretrained(args.model_name_or_path, args=args, attn_implementation="flash_attention_2")

    if 'phi' in args.model_type:
        target_modules = ['fc1', 'fc2'] if args.only_lora_ffn=='yes' else find_all_linear_names(model)
    elif 'stablelm' in args.model_type:
        target_modules = ['up_proj', 'down_proj', 'gate_proj'] if args.only_lora_ffn=='yes' else find_all_linear_names(model)

    if args.setting == "lora":
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias='none',
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer, image_processor, args)

    train_loop(args, data_module, model, tokenizer)