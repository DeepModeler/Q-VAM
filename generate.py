import os
import glob
import json
import argparse

import torch
from transformers import AutoTokenizer, CLIPImageProcessor

from models.class_arch import VQAModelphi, VQAModelstablelm
from models.base_arch import VQAbaseModelphi, VQAbaseModelstablelm
from dataset.vqadataset_test import VQAdataset


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

    parser.add_argument("--test_mode", type=str, default="sft", choices=("sft", "final"))
    parser.add_argument("--save_dir", default="./Q-VAM/save_model")
    parser.add_argument("--save_result", default="./Q-VAM/save_result")
    parser.add_argument("--data_path", default="./Q-VAM/VQA_datasets/train_all.json")
    parser.add_argument("--test_path", default="./Q-VAM/VQA_datasets")
    parser.add_argument("--image_folder", default="./Q-VAM/VQA_datasets/images") #
    parser.add_argument("--model_name_or_path", type=str, default="./Q-VAM/checkpoint")
    parser.add_argument("--image_tower_path", type=str, default="./Q-VAM/checkpoint/openai_clip-vit-large-patch14-336")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--mm_vision_select_feature", type=str, default="patch")
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--tokenizer_padding_side", type=str, default="right") #right left
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="./Q-VAM/checkpoint/base")
    parser.add_argument("--base_lm_model_path", type=str, default="./Q-VAM/checkpoint")
    parser.add_argument("--preK", type=int, default=4)
    parser.add_argument("--ablation", type=str, default="all", choices=("all", "w_SA", "w_CA"))

    # lora configuration
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--only_lora_ffn", type=str, default='no') #yes no

    args = parser.parse_args()
    return args

# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda:0'
    args = parse_argument()
    args.model_name_or_path = os.path.join(args.model_name_or_path, args.model_type)
    args.pretrain_mm_mlp_adapter = os.path.join(args.pretrain_mm_mlp_adapter, args.model_type, 'mm_projector.bin')
    image_processor = CLIPImageProcessor.from_pretrained(args.image_tower_path)

    #----------------model----------------
    if args.test_mode =='sft':
        if args.model_type == 'phi':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                      model_max_length=args.model_max_length,
                                                      padding_side="right",
                                                      use_fast=False, )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
            model = VQAbaseModelphi.from_pretrained(args.model_name_or_path, args=args)

        elif args.model_type == 'stablelm':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                      model_max_length=args.model_max_length,
                                                      padding_side="right",
                                                      use_fast=False, )
            tokenizer.unk_token = '<|reg0|>'
            model = VQAbaseModelstablelm.from_pretrained(args.model_name_or_path, args=args)

    elif args.test_mode =='final':
        final_model_path = os.path.join(args.save_dir, args.model_type + '_preK=' + str(args.preK) + '_abl' + args.ablation)
        pattern = os.path.join(final_model_path, 'checkpoint*')
        matched_paths = glob.glob(pattern)
        final_model_path = matched_paths[0]
        print('final_model_path: ', final_model_path)

        if args.model_type == 'phi':
            tokenizer = AutoTokenizer.from_pretrained(final_model_path,
                                                      model_max_length=args.model_max_length,
                                                      padding_side="right",
                                                      use_fast=False, )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
            model = VQAModelphi.from_pretrained(final_model_path, args=args)

        elif args.model_type == 'stablelm':
            tokenizer = AutoTokenizer.from_pretrained(final_model_path,
                                                      model_max_length=args.model_max_length,
                                                      padding_side="right",
                                                      use_fast=False, )
            tokenizer.unk_token = '<|reg0|>'
            model = VQAModelstablelm.from_pretrained(final_model_path, args=args)

    args.tokenizer_model_max_length = tokenizer.model_max_length
    print(args)

    model.to(device)

    if args.test_mode=='final':
        answers_file = os.path.join(args.save_result, args.dataset, args.test_mode, args.model_type + '_preK=' + str(args.preK) + '_abl_' + args.ablation + '.json')
    else:
        answers_file = os.path.join(args.save_result, args.dataset, args.test_mode, args.model_type + '.json')
    print(answers_file)
    ans_file = open(answers_file, "w")

    test_data_path = os.path.join(args.test_path, 'test_' + args.dataset + '.json')
    print('test_data_path: ', test_data_path)
    test_dataset = VQAdataset(data_path=test_data_path, tokenizer=tokenizer, image_processor=image_processor, data_args=args)


    for input_ids, image, ans_type, sample_id, question, ground_truth_answer, stopping_criteria in test_dataset:
        input_ids = input_ids.to(device)
        images = image.to(device)
        attention_masks = torch.ones(input_ids.size(1)).bool().to(device)

        if ans_type.lower()=="closed":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=2048,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=[stopping_criteria],
                )
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        elif ans_type.lower()=="open":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=2048,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        print('question: ', repr(question))
        print('ground truth answer: ', ground_truth_answer)
        print('predicted answer: ', outputs)

        ans_file.write(json.dumps({"sample_id": sample_id,
                                   "answer type": ans_type,
                                   "question": question,
                                   "predicted answer": outputs,
                                   "ground truth answer": ground_truth_answer,
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()