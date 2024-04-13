# pylint: disable=protected-access,too-many-arguments
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel
from transformers import BertModel, RobertaModel, AlbertModel, ElectraModel, XLMRobertaModel
from transformers import ViTMAEModel, PerceiverModel, GraphormerModel
import timm
from timm.models import VisionTransformer, Beit, checkpoint_seq
from timm.layers import PatchDropout

from .checkpoints import HF_CHECKPOINTS, TIMM_CHECKPOINTS

TextTransformer = (BertModel, RobertaModel, AlbertModel, ElectraModel, XLMRobertaModel)
TODO = (ViTMAEModel, PerceiverModel, GraphormerModel)


def setup_backbone(backbone, pretrained=True):
    """Initialize backbone architecture and load pretrained weights"""
    if backbone in HF_CHECKPOINTS:
        if pretrained:
            return AutoModel.from_pretrained(backbone)
        return AutoModel.from_config(AutoConfig.from_pretrained(backbone))
    if backbone in TIMM_CHECKPOINTS:
        return timm.create_model(backbone, pretrained=pretrained)
    raise NotImplementedError(f"Backbone ({backbone}) not supported!")


class Backbone(nn.Module):
    """Backbone transformer architecture (optionally pre-trained)"""
    def __init__(self, backbone, pretrained, patch_dropout, max_patch_dropout):
        super().__init__()
        assert (patch_dropout is None) or (max_patch_dropout is None)
        self.patch_drop_rate = patch_dropout
        self.max_patch_drop_rate = max_patch_dropout
        self.patch_drop = nn.Identity()
        self.backbone = setup_backbone(backbone, pretrained)
        if isinstance(self.backbone, TextTransformer):
            self.num_tokens = self.backbone.config.max_position_embeddings
            self.hidden_size = self.backbone.config.hidden_size
            self._forward = self._forward_text
            if patch_dropout or max_patch_dropout:
                raise NotImplementedError("PatchDropout not supported for TextTransformer!")
        elif isinstance(self.backbone, VisionTransformer):
            self.num_tokens = self.backbone.patch_embed.num_patches
            self.hidden_size = self.backbone.num_features
            self._forward = self._forward_vision
        elif isinstance(self.backbone, Beit):
            self.num_tokens = self.backbone.patch_embed.num_patches
            self.hidden_size = self.backbone.num_features
            self._forward = self._forward_beit
            if patch_dropout or max_patch_dropout:
                raise NotImplementedError("PatchDropout not supported for Beit!")
        elif isinstance(self.backbone, TODO):
            raise NotImplementedError(f"Backbone ({self.backbone}) not implemented (TODO)!")
        else:
            raise NotImplementedError(f"Backbone ({self.backbone}) not supported!")

    def _forward_text(self, x: torch.Tensor):
        output = self.backbone(inputs_embeds=x)
        unpooled_output = output.last_hidden_state
        pooled_output = output.pooler_output
        return unpooled_output, pooled_output

    def _forward_vision(self, x: torch.Tensor):
        # x = backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        if self.patch_drop_rate:
            self.patch_drop = PatchDropout(
                prob=self.patch_drop_rate,
                num_prefix_tokens=self.backbone.num_prefix_tokens)
        elif self.max_patch_drop_rate:
            self.patch_drop = PatchDropout(
                prob=torch.rand(1).item() * self.max_patch_drop_rate,
                num_prefix_tokens=self.backbone.num_prefix_tokens)
        else:
            assert self.patch_drop_rate is None and self.max_patch_drop_rate is None
            self.patch_drop = nn.Identity()
        x = self.patch_drop(x)
        x = self.backbone.norm_pre(x)
        if self.backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.backbone.blocks, x)
        else:
            x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        unpooled_output = x[:, self.backbone.num_prefix_tokens:]
        pooled_output = self.backbone.forward_head(x, pre_logits=True)
        return unpooled_output, pooled_output

    def _forward_beit(self, x: torch.Tensor):
        # x = self.backbone.patch_embed(x)
        b = x.size(0)
        x = torch.cat((self.backbone.cls_token.expand(b, -1, -1), x), dim=1)
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        rel_pos_bias = self.backbone.rel_pos_bias() if \
            self.backbone.rel_pos_bias is not None else None
        for blk in self.backbone.blocks:
            if self.backbone.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        x = self.backbone.norm(x)
        unpooled_output = x[:, self.backbone.num_prefix_tokens:]
        pooled_output = self.backbone.forward_head(x, pre_logits=True)
        return unpooled_output, pooled_output

    def forward(self, x: torch.Tensor):
        """forward pass for the backbone using features as input and output.
        bypass tokenizer and input embeddings, do not bypass positional embeddings.
        bypass global pooling and head classifier.
        :param x: input tensor of size (N, L, C)
        :return: output tensor of size (N, L, D) and pooled output of size (N, D)
        """
        assert x.dim() == 3, f"input tensor of size ({x.size()}) must be 3D (NLC)!"
        return self._forward(x)
