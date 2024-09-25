from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['alexnet_lrm3_63ab_pass2'] = lambda: ModelCommitment(
    identifier='alexnet_lrm3_63ab_pass2',
    activations_model=get_model(),
    layers=LAYERS)
