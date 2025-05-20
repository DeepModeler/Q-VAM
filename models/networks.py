import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F


class Inject_SA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, image_feat, text_feat):
        # Spatial attention via cross-attn
        attn_scores = torch.matmul(text_feat, image_feat.transpose(-2, -1))
        attn_weights = attn_scores.softmax(dim=-1)
        spatial_attn = attn_weights.mean(dim=0)

        image_feat = image_feat * (1 + self.alpha * spatial_attn.unsqueeze(1))
        return image_feat


class Inject_CA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_feat, text_feat):
        # Channel attention
        text_mean = text_feat.mean(dim=0).unsqueeze(0)
        channel_attn = F.sigmoid(self.channel_mlp(text_mean))
        image_feat = image_feat * channel_attn
        return image_feat


class Inject_SACA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, image_feat, text_feat):
        image_feat1 = image_feat.clone()
        # Channel attention
        text_mean = text_feat.mean(dim=0).unsqueeze(0)
        channel_attn = F.sigmoid(self.channel_mlp(text_mean))
        image_feat = image_feat * channel_attn

        # Spatial attention via cross-attn
        attn_scores = torch.matmul(text_feat, image_feat1.transpose(-2, -1))
        attn_weights = attn_scores.softmax(dim=-1)
        spatial_attn = attn_weights.mean(dim=0)

        image_feat = image_feat * (1 + self.alpha * spatial_attn.unsqueeze(1))
        return image_feat


class Linear_proj(nn.Module):
    def __init__(self, config, mlp_depth):
        super(Linear_proj, self).__init__()

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.image_spatial_proj = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_spatial_proj(x)


class CLIPVisionTower(nn.Module):
    def __init__(self, args, tune=False):
        super().__init__()
        self.image_tower_name = args.image_tower_path
        self.select_layer = args.mm_vision_select_layer #-2
        self.select_feature = args.mm_vision_select_feature #patch

        self.image_tower = CLIPVisionModel.from_pretrained(self.image_tower_name)
        if tune:
            config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.image_tower = get_peft_model(self.image_tower, config)
        else:
            # self.image_tower.requires_grad_(False)
            for p in self.image_tower.parameters():
                p.requires_grad = False

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad() 
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.image_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.dtype

    @property
    def device(self):
        return self.image_tower.device

    @property
    def config(self):
        return self.image_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2