
import copy
import torch.nn as nn

from .quantized_layer import (
    QuantizedLinear,
    QuantizedConv2d
)


def quantize_model(
    model,
    bits=8,
    granularity="per_channel",
    group_size=128,
    axis=0
):

    quantized_model = copy.deepcopy(model)

    def replace_layers(parent):

        for name, child in list(parent.named_children()):

            if isinstance(child, nn.Linear):

                quantized_layer = QuantizedLinear.from_float(
                    child,
                    bits=bits,
                    granularity=granularity,
                    group_size=group_size,
                    axis=axis
                )

                setattr(parent, name, quantized_layer)

            elif isinstance(child, nn.Conv2d):

                quantized_layer = QuantizedConv2d.from_float(
                    child,
                    bits=bits,
                    granularity=granularity,
                    group_size=group_size,
                    axis=axis
                )

                setattr(parent, name, quantized_layer)

            else:
                replace_layers(child)

    replace_layers(quantized_model)

    return quantized_model
