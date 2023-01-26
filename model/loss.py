import torch
import torch.nn.functional as F
import torch.autograd as autograd

from .layer_utils import CONV3_3_IN_VGG_19


def perceptual_loss(predicted, ground_truth):
    model = CONV3_3_IN_VGG_19

    # feature map of the output and target
    predicted_feature_map = model.forward(predicted)
    ground_truth_feature_map = model.forward(ground_truth).detach()  # we do not need the gradient of it
    loss = F.mse_loss(predicted_feature_map, ground_truth_feature_map)
    return loss

def pixel_loss(predicted, ground_truth):
    loss = F.l1_loss(predicted, ground_truth)
    return loss