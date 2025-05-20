from typing import List, Optional, Tuple, Union

import torch

from transformers import StableLmModel, StableLmForCausalLM
from transformers import PhiModel, PhiForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from models.networks import Linear_proj, CLIPVisionTower, Inject_CA, Inject_SA, Inject_SACA


'''============================================phi======================================='''
class ArchModelphi(PhiModel):
    def __init__(self, config, args):
        super(ArchModelphi, self).__init__(config)
        self.config = config
        self.args = args
        self.image_tower = CLIPVisionTower(args)
        self.mm_projector = Linear_proj(config, 2)
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        if args.ablation=='all':
            self.inject = Inject_SACA(config.hidden_size)
        elif args.ablation=='w_SA':
            self.inject = Inject_SA(config.hidden_size)
        elif args.ablation=='w_CA':
            self.inject = Inject_CA(config.hidden_size)

    def encode_images(self, images):
        image_features = self.image_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def context_fusion(self, device, prompt_len, inputs_embeds, label, attention_mask, position_ids, past_key_values=None):
        torch.autograd.set_detect_anomaly(True)
        context_fusion_layer = self.layers[:self.args.preK]
        output_attentions = self.config.output_attentions
        cache_position = torch.arange(0, 0 + inputs_embeds.shape[1], device=device)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        inputs_embeds = self.embed_dropout(inputs_embeds)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in context_fusion_layer:
            layerout = layer(hidden_states=hidden_states, attention_mask=causal_mask, position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions, position_embeddings=position_embeddings, attn_implementation="flash_attention_2")
            hidden_states = layerout[0]

        hidden_states = self.final_layernorm(hidden_states)

        maskt = (label != IMAGE_TOKEN_INDEX).to(device)
        maski = (label== IMAGE_TOKEN_INDEX).to(device)

        outputs_embeds = inputs_embeds[0].clone()
        fusion_text_embeds = hidden_states[0][maskt]
        image_embeds = inputs_embeds[0][maski]

        new_image_embeds = self.inject(image_embeds, fusion_text_embeds)
        idx = torch.where(maski == True)[0].to(device)
        outputs_embeds[idx] = new_image_embeds

        outputs_embeds = outputs_embeds[prompt_len:]
        outputs_label = label[prompt_len:]

        idx1 = torch.where(outputs_label==IMAGE_TOKEN_INDEX)[0].to(device)
        outputs_label[idx1] = IGNORE_INDEX
        return outputs_embeds, outputs_label


    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, labels, images):
        device = input_ids.device
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []
        image_features = self.encode_images(images_minibatch) # tensor

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cur_input_embeds = self.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=device, dtype=cur_labels.dtype))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # ----------------------------------------------------------------------------
            prompt_len = split_sizes[0]
            inputs_embeds_in = cur_new_input_embeds.unsqueeze(0)
            label_in = cur_new_labels.clone()
            attention_mask_in = torch.ones(1, inputs_embeds_in.shape[1], device=device).bool()
            position_ids_in = torch.arange(0, inputs_embeds_in.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            outputs_embeds, outputs_label = self.context_fusion(device, prompt_len, inputs_embeds_in, label_in, attention_mask_in, position_ids_in)
            # ----------------------------------------------------------------------------
            new_input_embeds.append(outputs_embeds)
            new_labels.append(outputs_label)

        tokenizer_model_max_length = self.args.tokenizer_model_max_length
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if self.args.tokenizer_padding_side == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
            else: #pad right
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return position_ids, attention_mask, new_input_embeds, new_labels



class VQAModelphi(PhiForCausalLM):
    def __init__(self, config, args):
        super(VQAModelphi, self).__init__(config)
        self.model = ArchModelphi(config, args)
        self.post_init()

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False, #None
        output_attentions: Optional[bool] = False, #None
        output_hidden_states: Optional[bool] = False, #None
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:

        position_ids1, attention_mask1, input_embeds, new_labels = self.model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, labels, images)

        outputs = super().forward(input_ids = None,
            attention_mask = attention_mask1,
            position_ids = position_ids1,
            past_key_values = past_key_values,
            inputs_embeds = input_embeds,
            labels = new_labels,
            use_cache=use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )
        return outputs




'''============================================stableLM======================================='''

class ArchModelstablelm(StableLmModel):
    def __init__(self, config, args):
        super(ArchModelstablelm, self).__init__(config)
        self.config = config
        self.args = args
        self.image_tower = CLIPVisionTower(args)
        self.mm_projector = Linear_proj(config, 2)
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        if args.ablation=='all':
            self.inject = Inject_SACA(config.hidden_size)
        elif args.ablation=='w_SA':
            self.inject = Inject_SA(config.hidden_size)
        elif args.ablation=='w_CA':
            self.inject = Inject_CA(config.hidden_size)

    def encode_images(self, images):
        image_features = self.image_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def context_fusion(self, device, prompt_len, inputs_embeds, label, attention_mask, position_ids, past_key_values=None, orig_image=None, tag=None):
        context_fusion_layer = self.layers[:self.args.preK]
        output_attentions = self.config.output_attentions
        cache_position = torch.arange(0, 0 + inputs_embeds.shape[1], device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in context_fusion_layer:
            layerout = layer(hidden_states=hidden_states, attention_mask=causal_mask, position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions, position_embeddings=position_embeddings)
            hidden_states = layerout[0]
        hidden_states = self.norm(hidden_states)

        maskt = (label != IMAGE_TOKEN_INDEX).to(device)
        maski = (label== IMAGE_TOKEN_INDEX).to(device)

        outputs_embeds = inputs_embeds[0].clone()
        fusion_text_embeds = hidden_states[0][maskt]
        image_embeds = inputs_embeds[0][maski]
        new_image_embeds = self.inject(image_embeds, fusion_text_embeds)
        idx = torch.where(maski == True)[0].to(device)
        outputs_embeds[idx] = new_image_embeds

        outputs_embeds = outputs_embeds[prompt_len:]
        outputs_label = label[prompt_len:]

        idx1 = torch.where(outputs_label==IMAGE_TOKEN_INDEX)[0].to(device)
        outputs_label[idx1] = IGNORE_INDEX

        return outputs_embeds, outputs_label


    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, labels, images):
        device = input_ids.device
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []
        image_features = self.encode_images(images_minibatch)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cur_input_embeds = self.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=device, dtype=cur_labels.dtype))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # ----------------------------------------------------------------------------
            prompt_len = split_sizes[0]
            inputs_embeds_in = cur_new_input_embeds.unsqueeze(0)
            label_in = cur_new_labels
            attention_mask_in = torch.ones(1, inputs_embeds_in.shape[1], device=device).bool()
            position_ids_in = torch.arange(0, inputs_embeds_in.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            outputs_embeds, outputs_label = self.context_fusion(device, prompt_len, inputs_embeds_in, label_in, attention_mask_in, position_ids_in)
            # ----------------------------------------------------------------------------
            new_input_embeds.append(outputs_embeds)
            new_labels.append(outputs_label)

        tokenizer_model_max_length = self.args.tokenizer_model_max_length
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if self.args.tokenizer_padding_side == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
            else: #pad right
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return position_ids, attention_mask, new_input_embeds, new_labels


class VQAModelstablelm(StableLmForCausalLM):
    def __init__(self, config, args):
        super(VQAModelstablelm, self).__init__(config)
        self.model = ArchModelstablelm(config, args)
        self.post_init()

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:

        position_ids1, attention_mask1, input_embeds, new_labels = self.model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, labels, images)

        outputs = super().forward(input_ids = None,
            attention_mask = attention_mask1,
            position_ids = position_ids1,
            past_key_values = past_key_values,
            inputs_embeds = input_embeds,
            labels = new_labels,
            use_cache=use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        ) 
        return outputs