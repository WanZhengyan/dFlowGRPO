"""
Differentiable CFG wrapper for training.

Two modes:
  - CFG mode (cfg_scale > 0): two forward passes (cond + uncond), combined via
        result = (1 + cfg_scale) * cond - cfg_scale * uncond
  - No-CFG mode (cfg_scale == 0): single forward pass, just returns cond logits.
        This halves the model forwards and can significantly speed up training.

We skip token_drop by temporarily setting model.training = False during the
forward pass. token_drop is FUDOKI's built-in 10% text-dropout for CFG training;
we don't need it because we either do CFG ourselves (explicitly masking text
for the uncond forward) or don't use CFG at all.
"""
import torch
from torch.nn.modules import Module
from flow_matching.utils import ModelWrapper


class DifferentiableCFGScaledModel(ModelWrapper):
    """
    Same interface as CFGScaledModel:
        model(x=x_t, cfg_scale=..., datainfo=...)
        -> (txt_result, img_result, datainfo)

    but WITHOUT torch.no_grad(), so p_theta is differentiable.

    When cfg_scale == 0, only one model forward is performed (no uncond pass).
    """

    def __init__(self, model: Module, g_or_u: str = "generation"):
        super().__init__(model)
        self.nfe_counter = 0
        self.g_or_u = g_or_u
        assert self.g_or_u in [
            "understanding",
            "generation",
            "generation and understanding",
        ]

    def forward(
        self, x: torch.Tensor, cfg_scale: float, datainfo, uncond_id=100015
    ):
        # --- NO torch.no_grad() here (the only difference from CFGScaledModel) ---

        # Temporarily disable training mode so token_drop is NOT called inside
        # MultiModalityCausalLM.forward().  We restore afterwards.
        was_training = self.model.training
        self.model.training = False  # skip token_drop; lighter than model.eval()

        conditional_img_logits, conditional_txt_logits = self.model(x, datainfo)

        if self.g_or_u == "generation":
            if cfg_scale > 0:
                # CFG: need a second (unconditional) forward pass
                uncondition_x = x.clone()
                text_token_mask = datainfo["text_token_mask"]
                for bs in range(text_token_mask.shape[0]):
                    nz = datainfo["text_token_mask"][bs].nonzero()
                    if nz.numel() > 0:
                        text_nonzero_idx_begin = nz[0, 0]
                        text_nonzero_idx_end = nz[-1, 0]
                        uncondition_x[
                            bs, text_nonzero_idx_begin : text_nonzero_idx_end + 1
                        ] = uncond_id
                unconditional_img_logits, _ = self.model(uncondition_x, datainfo)

                result_img = (
                    (1.0 + cfg_scale) * conditional_img_logits
                    - cfg_scale * unconditional_img_logits
                )
            else:
                # No CFG: single forward, just use conditional logits directly
                result_img = conditional_img_logits
            result_txt = None

        elif self.g_or_u == "understanding":
            from fudoki.eval_loop import top_k_logits
            result_txt = top_k_logits(conditional_txt_logits, top_k=1)
            result_img = None

        else:
            result_img = conditional_img_logits
            result_txt = conditional_txt_logits

        # Restore training mode flag
        self.model.training = was_training

        self.nfe_counter += 1

        if self.g_or_u == "understanding":
            return (
                torch.softmax(result_txt.to(dtype=torch.float32), dim=-1),
                result_img,
                datainfo,
            )
        elif self.g_or_u == "generation":
            return (
                result_txt,
                torch.softmax(result_img.to(dtype=torch.float32), dim=-1),
                datainfo,
            )
        else:
            return (
                torch.softmax(result_txt.to(dtype=torch.float32), dim=-1),
                torch.softmax(result_img.to(dtype=torch.float32), dim=-1),
                datainfo,
            )

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter
