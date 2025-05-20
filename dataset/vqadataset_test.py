import os
import json
from typing import Dict

import torch
from torch.utils.data import Dataset
from PIL import Image

from constants import IMAGE_TOKEN_INDEX
import conversation as conversation_lib
from conversation import conv_phi
from utils.mm_utils import tokenizer_image_token, expand2square, KeywordsStoppingCriteria



class VQAdataset(Dataset):
    def __init__(self, data_path: str, tokenizer, image_processor, data_args):
        super(VQAdataset, self).__init__()

        list_data_dict = []
        data = json.load(open(data_path, "r"))
        for i in data:
            i['id'] = str(len(list_data_dict))+ '_original_id='+str(i['id'])
            list_data_dict.append(i)

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        image = Image.open(os.path.join(self.data_args.image_folder, sources['image'])).convert('RGB')
        if self.data_args.image_aspect_ratio == 'pad':
            image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        current_conv = sources["conversations"]
        ans_type = sources["answer_type"]

        question = current_conv[0]["value"]
        ground_truth_answer = current_conv[1]["value"]

        conv = conv_phi.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        input_text = conv.get_prompt()

        input_ids = tokenizer_image_token(input_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        sample_id = sources['id']

        return input_ids, image, ans_type, sample_id, question, ground_truth_answer, stopping_criteria