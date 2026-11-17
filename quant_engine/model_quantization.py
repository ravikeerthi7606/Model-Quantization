
import copy
import torch
import torch.nn as nn

from .quantizer import Quantizer


def quantize_model(model, bits=8, granularity="per_channel",
                    axis=0):

    quantized_model = copy.deepcopy(model)

    for name, module in quantized_model.named_modules():

        if isinstance(module, (nn.Linear, nn.Conv2d)):

            quantizer = Quantizer(
                bits=bits,
                granularity=granularity,
                axis=axis
            )

            q_weights, params = quantizer.quantize(
                module.weight.data
            )

            reconstructed_weights = quantizer.dequantize(
                q_weights, params
            )

            with torch.no_grad():
                module.weight.copy_(reconstructed_weights)

    return quantized_model
