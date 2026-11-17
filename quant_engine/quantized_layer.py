
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import Quantizer
from .packing import pack_int4, unpack_int4


class QuantizedLinear(nn.Module):

    def __init__(
        self,
        weight,
        scales,
        bias,
        metadata,
        in_features,
        out_features
    ):
        super().__init__()

        # Store quantized weights as a registered buffer
        self.register_buffer("weight", weight)

        # Store quantization scales
        self.register_buffer("scales", scales)

        # Bias remains floating point
        self.register_buffer("bias", bias)

        self.metadata = metadata

        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_float(
        cls,
        layer,
        bits=8,
        granularity="per_channel",
        group_size=128,
        axis=0
    ):

        if not isinstance(layer, nn.Linear):
            raise TypeError("Expected an nn.Linear layer")

        quantizer = Quantizer(
            bits=bits,
            granularity=granularity,
            group_size=group_size,
            axis=axis
        )

        # Quantize original weights
        q_weight, params = quantizer.quantize(
            layer.weight.detach()
        )

        # Store scale separately from other metadata
        scales = params["scale"]

        # Prepare metadata
        metadata = {
            "bits": bits,
            "granularity": granularity,
            "axis": axis,
            "group_size": group_size,
            "original_shape": tuple(layer.weight.shape),
            "original_numel": layer.weight.numel(),
            "original_dtype": str(layer.weight.dtype),
            "packing": None
        }

        # Pack INT4 weights
        if bits == 4:

            packed_weight, packing_metadata = pack_int4(
                q_weight
            )

            weight = packed_weight
            metadata["packing"] = "int4"
            metadata["packing_metadata"] = packing_metadata

        else:
            weight = q_weight

        # Copy bias without modifying original layer
        bias = (
            layer.bias.detach().clone()
            if layer.bias is not None
            else None
        )

        return cls(
            weight=weight,
            scales=scales,
            bias=bias,
            metadata=metadata,
            in_features=layer.in_features,
            out_features=layer.out_features
        )

    def dequantize_weight(self):

        bits = self.metadata["bits"]

        # Recover INT4 values if weights are packed
        if bits == 4:

            q_weight = unpack_int4(
                self.weight,
                self.metadata["packing_metadata"]
            )

        else:
            q_weight = self.weight

        # Reconstruct quantization parameters
        params = {
            "scale": self.scales,
            "bits": bits,
            "granularity": self.metadata["granularity"]
        }

        if self.metadata["granularity"] == "per_channel":
            params["axis"] = self.metadata["axis"]

        elif self.metadata["granularity"] == "per_group":
            params["axis"] = self.metadata["axis"]
            params["group_size"] = self.metadata["group_size"]

        quantizer = Quantizer(
            bits=bits,
            granularity=self.metadata["granularity"],
            group_size=self.metadata["group_size"],
            axis=self.metadata["axis"]
        )

        return quantizer.dequantize(
            q_weight,
            params
        )

    def forward(self, x):

        # Reconstruct floating-point weights
        weight = self.dequantize_weight()

        # Match input dtype for floating-point computation
        weight = weight.to(dtype=x.dtype)

        bias = (
            self.bias.to(dtype=x.dtype)
            if self.bias is not None
            else None
        )

        return F.linear(
            x,
            weight,
            bias
        )



class QuantizedConv2d(nn.Module):

    def __init__(
        self,
        weight,
        scales,
        bias,
        metadata,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        padding_mode
    ):
        super().__init__()

        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)

        self.metadata = metadata

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

    @classmethod
    def from_float(
        cls,
        layer,
        bits=8,
        granularity="per_channel",
        group_size=128,
        axis=0
    ):

        if not isinstance(layer, nn.Conv2d):
            raise TypeError("Expected an nn.Conv2d layer")

        quantizer = Quantizer(
            bits=bits,
            granularity=granularity,
            group_size=group_size,
            axis=axis
        )

        # Quantize convolution weights
        q_weight, params = quantizer.quantize(
            layer.weight.detach()
        )

        scales = params["scale"]

        metadata = {
            "bits": bits,
            "granularity": granularity,
            "axis": axis,
            "group_size": group_size,
            "original_shape": tuple(layer.weight.shape),
            "original_numel": layer.weight.numel(),
            "original_dtype": str(layer.weight.dtype),
            "packing": None
        }

        # Pack INT4 weights
        if bits == 4:

            packed_weight, packing_metadata = pack_int4(
                q_weight
            )

            weight = packed_weight

            metadata["packing"] = "int4"
            metadata["packing_metadata"] = packing_metadata

        else:
            weight = q_weight

        bias = (
            layer.bias.detach().clone()
            if layer.bias is not None
            else None
        )

        return cls(
            weight=weight,
            scales=scales,
            bias=bias,
            metadata=metadata,
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            padding_mode=layer.padding_mode
        )

    def dequantize_weight(self):

        bits = self.metadata["bits"]

        if bits == 4:

            q_weight = unpack_int4(
                self.weight,
                self.metadata["packing_metadata"]
            )

        else:
            q_weight = self.weight

        params = {
            "scale": self.scales,
            "bits": bits,
            "granularity": self.metadata["granularity"]
        }

        if self.metadata["granularity"] == "per_channel":
            params["axis"] = self.metadata["axis"]

        elif self.metadata["granularity"] == "per_group":
            params["axis"] = self.metadata["axis"]
            params["group_size"] = self.metadata["group_size"]

        quantizer = Quantizer(
            bits=bits,
            granularity=self.metadata["granularity"],
            group_size=self.metadata["group_size"],
            axis=self.metadata["axis"]
        )

        return quantizer.dequantize(
            q_weight,
            params
        )

    def forward(self, x):

        weight = self.dequantize_weight()
        weight = weight.to(dtype=x.dtype)

        bias = (
            self.bias.to(dtype=x.dtype)
            if self.bias is not None
            else None
        )

        # Handle non-zero padding modes
        if self.padding_mode != "zeros":

            padding = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                padding=self.padding,
                padding_mode=self.padding_mode
            )._reversed_padding_repeated_twice

            x = F.pad(
                x,
                padding,
                mode=self.padding_mode
            )

            padding = 0

        else:
            padding = self.padding

        return F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups
        )
