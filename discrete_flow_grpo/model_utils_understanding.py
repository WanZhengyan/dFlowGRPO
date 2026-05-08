"""
Helpers for multimodal-understanding GRPO training.

Analogous to ``model_utils.build_data_info`` / ``utils.prompts_to_tensor`` but
specialized for the "understanding" direction (image + question -> text answer).
"""

import torch
from torchvision import transforms
from PIL import Image

from config import IMG_LEN
from utils import prompts_to_tensor


# Image preprocessing (same as FUDOKI/inference_i2t_local.py).
def _resize_pad(image, image_size=384):
    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    resize_scale = image_size / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))
    new_image = Image.new("RGB", (image_size, image_size), (127, 127, 127))
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    new_image.paste(image, ((image_size - new_w) // 2, (image_size - new_h) // 2))
    return new_image


_TFM = transforms.Compose([
    transforms.Lambda(_resize_pad),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])


def preprocess_image(pil_img):
    """PIL Image -> normalized tensor (3, 384, 384)."""
    return _TFM(pil_img.convert("RGB"))


def build_data_info_understanding(items, vl_chat_processor, path_txt,
                                  vocabulary_size_txt, txt_max_length, device):
    """
    Build (x_init, data_info) for a batch of understanding items.

    Args:
        items: list of dicts from ``ScienceQADataset`` (must contain ``prompt``
            and ``image`` — a PIL image or None).

    Returns:
        x_init: (B, L) LongTensor. Text token positions filled with uniform noise;
                all other positions hold the prompt / image-slot ids.
        data_info: dict with image_token_mask, text_token_mask, understanding_img, ...
    """
    prompts = [it["prompt"] for it in items]
    imgs = []
    for it in items:
        if it["image"] is not None:
            imgs.append(preprocess_image(it["image"]))
        else:
            imgs.append(None)

    B = len(items)
    # ``prompts_to_tensor`` for understanding only needs text t=0 (all-noise).
    t = torch.zeros(B, device=device)
    # Placeholder img_tokens (unused for understanding).
    dummy_img_tokens = torch.zeros(B, IMG_LEN, dtype=torch.long, device=device)

    try:
        input_ids_t, origin_ids, data_info = prompts_to_tensor(
            prompts=prompts, vl_chat_processor=vl_chat_processor,
            path=path_txt, t=t, g_or_u="understanding",
            txt_max_length=txt_max_length, IMG_LEN=IMG_LEN,
            img_in_question=imgs, img_tokens=dummy_img_tokens, device=device,
        )
    except ValueError as e:
        if "Split token not found" in str(e):
            qids = [it.get("qid", "?") for it in items]
            prompt_lens = [len(p) for p in prompts]
            raise ValueError(
                f"Prompt too long after truncation — 'Assistant:' got cut off. "
                f"qids={qids} prompt_char_lens={prompt_lens} "
                f"txt_max_length={txt_max_length}. "
                f"Tighten ``max_prompt_chars`` in ScienceQADataset or raise "
                f"``txt_max_length``. Original error: {e}"
            ) from e
        raise
    # input_ids_t already has text positions replaced by uniform noise (see utils.py,
    # the else branch when t==0).
    return input_ids_t, data_info
