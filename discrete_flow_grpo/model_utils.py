"""
Model loading, image decoding, data_info construction, and CFG wrappers.

Shared between training (train_grpo.py) and evaluation (evaluate_geneval.py).
"""

import torch
from peft import PeftModel

from fudoki.model import instantiate_model
from config import VOCABULARY_SIZE_IMG, IMG_LEN
from utils import prompts_to_tensor


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def decode_image_tokens(model, image_token_ids):
    """VQ tokens -> pixel images (B, 3, 384, 384) in [0, 1]."""
    B = image_token_ids.shape[0]
    try:
        dec_device = next(model.gen_vision_model.parameters()).device
    except StopIteration:
        dec_device = torch.device("cpu")
    image_token_ids = image_token_ids.to(dec_device)
    with torch.cuda.amp.autocast(enabled=False):
        images = model.gen_vision_model.decode_code(
            image_token_ids, [B, 8, 24, 24])
    return torch.clamp((images + 1) / 2.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# data_info construction
# ---------------------------------------------------------------------------
def build_data_info(prompt, batch_size, vl_chat_processor, path_img,
                    txt_max_length, device):
    """
    Build data_info dict and x_init for generation from a single text prompt.

    Returns: (x_init, data_info)
        x_init:    (B, L) token sequence with image positions filled with random noise
        data_info: dict with masks, attention_mask, etc.
    """
    prompts = [prompt] * batch_size
    dummy = torch.zeros(batch_size, IMG_LEN, dtype=torch.long, device=device)
    t = torch.zeros(batch_size, device=device)
    input_ids_t, _, data_info = prompts_to_tensor(
        prompts=prompts, vl_chat_processor=vl_chat_processor,
        path=path_img, t=t, g_or_u="generation",
        txt_max_length=txt_max_length, IMG_LEN=IMG_LEN,
        img_tokens=dummy, device=device,
    )
    img_mask = data_info["image_token_mask"]
    x_init = input_ids_t.clone()
    noise = torch.randint(0, VOCABULARY_SIZE_IMG,
                          (batch_size, IMG_LEN), dtype=torch.long, device=device)
    x_init[img_mask == 1] = noise.flatten()
    return x_init, data_info


def build_data_info_multi(prompt_list, G, vl_chat_processor, path_img,
                          txt_max_length, device):
    """
    Build data_info dict and x_init for multiple prompts, each repeated G times.
    Total batch size = len(prompt_list) * G.

    Returns: (x_init, data_info)
    """
    P = len(prompt_list)
    total_B = P * G
    prompts = []
    for p in prompt_list:
        prompts.extend([p] * G)

    dummy = torch.zeros(total_B, IMG_LEN, dtype=torch.long, device=device)
    t = torch.zeros(total_B, device=device)
    input_ids_t, _, data_info = prompts_to_tensor(
        prompts=prompts, vl_chat_processor=vl_chat_processor,
        path=path_img, t=t, g_or_u="generation",
        txt_max_length=txt_max_length, IMG_LEN=IMG_LEN,
        img_tokens=dummy, device=device,
    )
    img_mask = data_info["image_token_mask"]
    x_init = input_ids_t.clone()
    noise = torch.randint(0, VOCABULARY_SIZE_IMG,
                          (total_B, IMG_LEN), dtype=torch.long, device=device)
    x_init[img_mask == 1] = noise.flatten()
    return x_init, data_info


# ---------------------------------------------------------------------------
# Model loading (for evaluation — loads from saved checkpoints)
# ---------------------------------------------------------------------------
def load_model(checkpoint_path, lora_path=None, ema_path=None,
               gen_modules_path=None, device="cuda"):
    """Load FUDOKI base model + optional LoRA / EMA / gen_modules."""
    model = instantiate_model(checkpoint_path).to(device).to(torch.float32)

    if gen_modules_path is not None:
        gs = torch.load(gen_modules_path)
        ms = model.state_dict()
        for n, t in gs.items():
            if n in ms:
                ms[n].copy_(t.to(ms[n].device))
        model.load_state_dict(ms, strict=False)

    if lora_path is not None:
        model.language_model = PeftModel.from_pretrained(
            model.language_model, lora_path)
        model.language_model.eval()

        if ema_path is not None:
            es = torch.load(ema_path)
            ema_params = es["ema_parameters"]
            lora_ps = [p for n, p in model.language_model.named_parameters()
                       if "lora" in n.lower()]
            gen_ps = []
            if gen_modules_path is not None:
                gen_ps = [p for n, p in model.named_parameters()
                          if n.startswith(("gen_aligner.", "gen_head."))]
            all_t = lora_ps + gen_ps
            assert len(ema_params) == len(all_t), (
                f"EMA {len(ema_params)} != trainable {len(all_t)}")
            for ep, param in zip(ema_params, all_t):
                param.data.copy_(ep.to(param.device).data)

    model.eval()
    if hasattr(model, "gen_vision_model"):
        model.gen_vision_model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# No-CFG wrapper (for evaluation without classifier-free guidance)
# ---------------------------------------------------------------------------
class NoCFGModel:
    """Wrapper matching CFGScaledModel interface but skipping CFG (single forward pass)."""

    def __init__(self, model):
        self.model = model

    def __call__(self, x, cfg_scale, datainfo):
        with torch.no_grad():
            cond_img, cond_txt = self.model(x, datainfo)
        return cond_txt, torch.softmax(cond_img.to(torch.float32), dim=-1), datainfo

    def eval(self):
        self.model.eval()
        return self
