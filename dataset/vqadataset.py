import os
import json
import copy
import random

from dataclasses import dataclass
from typing import Sequence, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
import transformers

from constants import IGNORE_INDEX
import conversation as conversation_lib
from utils.mm_utils import tokenizer_image_token, expand2square


class VQAdataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, data_args):
        super(VQAdataset, self).__init__()

        list_data_dict = []
        data = json.load(open(data_path, "r"))
        for i in data:
            i['id'] = len(list_data_dict)
            list_data_dict.append(i)

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1

            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder

            image_file = image_file if isinstance(image_file, list) else [image_file]
            image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
            if self.data_args.image_aspect_ratio == 'pad':
                image = [expand2square(i, tuple(int(x * 255) for x in self.image_processor.image_mean)) for i in image]
                image = [self.image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
            else:
                image = [self.image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
            sources  = copy.deepcopy([e["conversations"] for e in sources])

            data_dict = preprocess_phi(sources, self.tokenizer)

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

            data_dict['image'] = image

            return data_dict

        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__()-1))


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids,
                     labels=labels,
                     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),)

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]

            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            batch['images'] = images
        else:
            raise ValueError(f'pretrain, {instances}')
        return batch


def preprocess_phi(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conv = conversation_lib.conv_templates['phi']
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."f" (ignored)")
    return dict(input_ids=input_ids, labels=targets,)


def make_supervised_data_module(tokenizer, image_processor, data_args) -> Dict:
    train_dataset = VQAdataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)