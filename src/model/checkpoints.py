# huggingface checkpoints
HF_CHECKPOINTS = [
    # text transformers
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'roberta-base',
    'roberta-large',
    'albert-base-v2',
    'albert-large-v2',
    'albert-xlarge-v2',
    'albert-xxlarge-v2',
    'google/electra-small-discriminator',
    'google/electra-base-discriminator',
    'google/electra-large-discriminator',
    'xlm-roberta-base',
    'xlm-roberta-large',
    # MAE pretrained
    'facebook/vit-mae-base',
    'facebook/vit-mae-large',
    'facebook/vit-mae-huge',
    # CoCa (does not work)
    # 'laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k',
    # 'laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k',
    # 'laion/CoCa-ViT-B-32-laion2B-s13B-b90k',
    # 'laion/CoCa-ViT-L-14-laion2B-s13B-b90k',
    # Perceiver and Perceiver IO
    'deepmind/language-perceiver',
    'deepmind/multimodal-perceiver',
    'deepmind/optical-flow-perceiver',
    'deepmind/vision-perceiver-learned',
    'deepmind/vision-perceiver-fourier',
    'deepmind/vision-perceiver-conv',
    # Graphormer
    'clefourrier/graphormer-base-pcqm4mv2'
]

# timm checkpoints
TIMM_CHECKPOINTS = [
    'timm/beit_base_patch16_224.in22k_ft_in22k',  # TODO: temporary
    # re-finetuned augreg 21k FT on in1k weights
    'hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch16_384.augreg2_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch8_224.augreg2_in21k_ft_in1k',
    # How to train your ViT (augreg) weights, pretrained on 21k FT on in1k
    'hf_hub:timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_small_patch32_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_small_patch32_384.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_small_patch16_384.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch32_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch32_384.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch16_384.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_large_patch16_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_large_patch16_224.augreg_in21k_ft_in1k',
    'hf_hub:timm/vit_large_patch16_384.augreg_in21k_ft_in1k',
    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    'hf_hub:timm/vit_base_patch16_224.orig_in21k_ft_in1k',
    'hf_hub:timm/vit_base_patch16_384.orig_in21k_ft_in1k',
    'hf_hub:timm/vit_large_patch32_384.orig_in21k_ft_in1k',
    # How to train your ViT (augreg) weights trained on in1k only
    'hf_hub:timm/vit_small_patch16_224.augreg_in1k',
    'hf_hub:timm/vit_small_patch16_384.augreg_in1k',
    'hf_hub:timm/vit_base_patch32_224.augreg_in1k',
    'hf_hub:timm/vit_base_patch32_384.augreg_in1k',
    'hf_hub:timm/vit_base_patch16_224.augreg_in1k',
    'hf_hub:timm/vit_base_patch16_384.augreg_in1k',
    # patch models, imagenet21k (weights from official Google JAX impl)
    'hf_hub:timm/vit_large_patch32_224.orig_in21k',
    'hf_hub:timm/vit_huge_patch14_224.orig_in21k',
    # How to train your ViT (augreg) weights, pretrained on in21k
    'hf_hub:timm/vit_tiny_patch16_224.augreg_in21k',
    'hf_hub:timm/vit_small_patch32_224.augreg_in21k',
    'hf_hub:timm/vit_small_patch16_224.augreg_in21k',
    'hf_hub:timm/vit_base_patch32_224.augreg_in21k',
    'hf_hub:timm/vit_base_patch16_224.augreg_in21k',
    'hf_hub:timm/vit_base_patch8_224.augreg_in21k',
    'hf_hub:timm/vit_large_patch16_224.augreg_in21k',
    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'hf_hub:timm/vit_base_patch32_224.sam',
    'hf_hub:timm/vit_base_patch16_224.sam',
    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'hf_hub:timm/vit_small_patch16_224.dino',
    'hf_hub:timm/vit_small_patch8_224.dino',
    'hf_hub:timm/vit_base_patch16_224.dino',
    'hf_hub:timm/vit_base_patch8_224.dino',
    # ViT ImageNet-21K-P pretraining by MILL
    'hf_hub:timm/vit_base_patch16_224_miil.in21k',
    'hf_hub:timm/vit_base_patch16_224_miil.in21k_ft_in1k',
    # BEiT fine-tuned weights from MAE style MIM - BEiT target pretrain
    'hf_hub:timm/beit_base_patch16_224.in22k_ft_in22k',
    'hf_hub:timm/beit_base_patch16_224.in22k_ft_in22k_in1k',
    'hf_hub:timm/beit_base_patch16_384.in22k_ft_in22k_in1k',
    'hf_hub:timm/beit_large_patch16_224.in22k_ft_in22k',
    'hf_hub:timm/beit_large_patch16_224.in22k_ft_in22k_in1k',
    'hf_hub:timm/beit_large_patch16_384.in22k_ft_in22k_in1k',
    'hf_hub:timm/beit_large_patch16_512.in22k_ft_in22k_in1k',
    'hf_hub:timm/beitv2_base_patch16_224.in1k_ft_in22k',
    'hf_hub:timm/beitv2_base_patch16_224.in1k_ft_in22k_in1k',
    'hf_hub:timm/beitv2_large_patch16_224.in1k_ft_in22k',
    'hf_hub:timm/beitv2_large_patch16_224.in1k_ft_in22k_in1k',
    # DeiT models (FB weights)
    'hf_hub:timm/deit_tiny_patch16_224.fb_in1k',
    'hf_hub:timm/deit_small_patch16_224.fb_in1k',
    'hf_hub:timm/deit_base_patch16_224.fb_in1k',
    'hf_hub:timm/deit_base_patch16_384.fb_in1k',
    'hf_hub:timm/deit_tiny_distilled_patch16_224.fb_in1k',
    'hf_hub:timm/deit_small_distilled_patch16_224.fb_in1k',
    'hf_hub:timm/deit_base_distilled_patch16_224.fb_in1k',
    'hf_hub:timm/deit_base_distilled_patch16_384.fb_in1k',
    'hf_hub:timm/deit3_small_patch16_224.fb_in1k',
    'hf_hub:timm/deit3_small_patch16_384.fb_in1k',
    'hf_hub:timm/deit3_medium_patch16_224.fb_in1k',
    'hf_hub:timm/deit3_base_patch16_224.fb_in1k',
    'hf_hub:timm/deit3_base_patch16_384.fb_in1k',
    'hf_hub:timm/deit3_large_patch16_224.fb_in1k',
    'hf_hub:timm/deit3_large_patch16_384.fb_in1k',
    'hf_hub:timm/deit3_huge_patch14_224.fb_in1k',
    'hf_hub:timm/deit3_small_patch16_224.fb_in22k_ft_in1k',
    'hf_hub:timm/deit3_small_patch16_384.fb_in22k_ft_in1k',
    'hf_hub:timm/deit3_medium_patch16_224.fb_in22k_ft_in1k',
    'hf_hub:timm/deit3_base_patch16_224.fb_in22k_ft_in1k',
    'hf_hub:timm/deit3_base_patch16_384.fb_in22k_ft_in1k',
    'hf_hub:timm/deit3_large_patch16_224.fb_in22k_ft_in1k',
    'hf_hub:timm/deit3_large_patch16_384.fb_in22k_ft_in1k',
    'hf_hub:timm/deit3_huge_patch14_224.fb_in22k_ft_in1k',
    # FlexiViT models
    'hf_hub:timm/flexivit_small.1200ep_in1k',
    'hf_hub:timm/flexivit_small.600ep_in1k',
    'hf_hub:timm/flexivit_small.300ep_in1k',
    'hf_hub:timm/flexivit_base.1200ep_in1k',
    'hf_hub:timm/flexivit_base.600ep_in1k',
    'hf_hub:timm/flexivit_base.300ep_in1k',
    'hf_hub:timm/flexivit_base.1000ep_in21k',
    'hf_hub:timm/flexivit_base.300ep_in21k',
    'hf_hub:timm/flexivit_large.1200ep_in1k',
    'hf_hub:timm/flexivit_large.600ep_in1k',
    'hf_hub:timm/flexivit_large.300ep_in1k',
    'hf_hub:timm/flexivit_base.patch16_in21k',
    'hf_hub:timm/flexivit_base.patch30_in21k',
    # CLIP pretrained image tower and related fine-tuned weights
    'hf_hub:timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch32_clip_384.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_large_patch14_clip_224.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch32_clip_224.openai_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch32_clip_384.openai_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch32_clip_448.openai_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch16_clip_224.openai_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch16_clip_384.openai_ft_in12k_in1k',
    'hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k',
    'hf_hub:timm/vit_large_patch14_clip_336.openai_ft_in12k_in1k',
    'hf_hub:timm/vit_base_patch32_clip_224.laion2b_ft_in1k',
    'hf_hub:timm/vit_base_patch16_clip_224.laion2b_ft_in1k',
    'hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in1k',
    'hf_hub:timm/vit_large_patch14_clip_224.laion2b_ft_in1k',
    'hf_hub:timm/vit_large_patch14_clip_336.laion2b_ft_in1k',
    'hf_hub:timm/vit_huge_patch14_clip_224.laion2b_ft_in1k',
    'hf_hub:timm/vit_huge_patch14_clip_336.laion2b_ft_in1k',
    'hf_hub:timm/vit_base_patch32_clip_224.openai_ft_in1k',
    'hf_hub:timm/vit_base_patch16_clip_224.openai_ft_in1k',
    'hf_hub:timm/vit_base_patch16_clip_384.openai_ft_in1k',
    'hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in1k',
    'hf_hub:timm/vit_base_patch32_clip_224.laion2b_ft_in12k',
    'hf_hub:timm/vit_base_patch16_clip_224.laion2b_ft_in12k',
    'hf_hub:timm/vit_large_patch14_clip_224.laion2b_ft_in12k',
    'hf_hub:timm/vit_huge_patch14_clip_224.openai_ft_in12k',
    'hf_hub:timm/vit_base_patch32_clip_224.openai_ft_in12k',
    'hf_hub:timm/vit_base_patch16_clip_224.openai_ft_in12k',
    'hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k',
    'hf_hub:timm/vit_base_patch32_clip_224.laion2b',
    'hf_hub:timm/vit_base_patch16_clip_224.laion2b',
    'hf_hub:timm/vit_large_patch14_clip_224.laion2b',
    'hf_hub:timm/vit_huge_patch14_clip_224.laion2b',
    'hf_hub:timm/vit_giant_patch14_clip_224.laion2b',
    'hf_hub:timm/vit_gigantic_patch14_clip_224.laion2b',
    'hf_hub:timm/vit_base_patch32_clip_224.openai',
    'hf_hub:timm/vit_base_patch16_clip_224.openai',
    'hf_hub:timm/vit_large_patch14_clip_224.openai',
    'hf_hub:timm/vit_large_patch14_clip_336.openai',
    # EVA fine-tuned weights from MAE style MIM - EVA-CLIP target pretrain
    'hf_hub:timm/eva_large_patch14_196.in22k_ft_in22k_in1k',
    'hf_hub:timm/eva_large_patch14_336.in22k_ft_in22k_in1k',
    'hf_hub:timm/eva_large_patch14_196.in22k_ft_in1k',
    'hf_hub:timm/eva_large_patch14_336.in22k_ft_in1k',
]
