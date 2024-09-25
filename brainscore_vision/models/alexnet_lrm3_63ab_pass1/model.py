import functools

import torch
import torchvision.models
import torch.nn as nn

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """

"""

LAYERS = ['model.feedforward.features.2', 'model.feedforward.features.5', 'model.feedforward.features.7', 'model.feedforward.features.9', 
          'model.feedforward.features.12', 'model.feedforward.classifier.2', 'model.feedforward.classifier.5']

class ModelWrapper(nn.Module):
    def __init__(self, model, forward_passes=2):
        super(ModelWrapper, self).__init__() 
        self.model = model
        self.forward_passes = forward_passes
    
    def forward(self, x):
        out = self.model(x, forward_passes=self.forward_passes)
        return out[-1]
        
def get_model():
    lrm, transforms = torch.hub.load('harvard-visionlab/lrm-steering', 'alexnet_lrm3', pretrained=True, steering=False, force_reload=False)
    model = ModelWrapper(lrm, forward_passes=2)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='alexnet_lrm3_63ab_pass1', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper
