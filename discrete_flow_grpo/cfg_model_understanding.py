"""
CFG wrappers for the understanding direction (text generation conditioned on image).

This module just provides thin aliases/subclasses of the generation-side CFG
wrappers from ``cfg_model.py`` and ``fudoki.eval_loop.CFGScaledModel`` with
``g_or_u='understanding'`` pre-set.

The interface is identical: ``model(x, cfg_scale, datainfo)`` returns
``(p_txt_softmax, None, datainfo)`` for understanding mode.
"""
from fudoki.eval_loop import CFGScaledModel
from cfg_model import DifferentiableCFGScaledModel


class UnderstandingCFGModel(CFGScaledModel):
    """No-grad CFG wrapper for understanding-direction sampling."""

    def __init__(self, model):
        super().__init__(model, g_or_u="understanding")


class DifferentiableUnderstandingCFGModel(DifferentiableCFGScaledModel):
    """Differentiable CFG wrapper for understanding-direction training."""

    def __init__(self, model):
        super().__init__(model, g_or_u="understanding")
