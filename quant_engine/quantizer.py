
from .per_tensor import (
    quantize_tensor,
    dequantize_tensor
)

from .per_channel import (
    quantize_per_channel,
    dequantize_per_channel
)

from .per_group import (
    quantize_per_group,
    dequantize_per_group
)


class Quantizer:

    def __init__(
        self,
        bits=8,
        scheme="symmetric",
        granularity="per_tensor",
        group_size=128,
        axis=0
    ):

        if bits != 8:
            raise NotImplementedError(
                "Currently only INT8 is supported."
            )

        if scheme != "symmetric":
            raise NotImplementedError(
                "Currently only symmetric quantization is supported."
            )

        if granularity not in [
            "per_tensor",
            "per_channel",
            "per_group"
        ]:
            raise ValueError("Invalid granularity")

        if group_size <= 0:
            raise ValueError("group_size must be positive")

        self.bits = bits
        self.scheme = scheme
        self.granularity = granularity
        self.group_size = group_size
        self.axis = axis

    def quantize(self, tensor):

        if self.granularity == "per_tensor":

            q_tensor, scale = quantize_tensor(tensor)

            params = {
                "scale": scale,
                "granularity": self.granularity
            }

        elif self.granularity == "per_channel":

            q_tensor, scale = quantize_per_channel(
                tensor,
                axis=self.axis
            )

            params = {
                "scale": scale,
                "axis": self.axis,
                "granularity": self.granularity
            }

        else:

            q_tensor, scale = quantize_per_group(
                tensor,
                group_size=self.group_size,
                axis=self.axis
            )

            params = {
                "scale": scale,
                "axis": self.axis,
                "group_size": self.group_size,
                "granularity": self.granularity
            }

        return q_tensor, params

    def dequantize(self, q_tensor, params):

        granularity = params["granularity"]
        scale = params["scale"]

        if granularity == "per_tensor":

            return dequantize_tensor(
                q_tensor,
                scale
            )

        elif granularity == "per_channel":

            return dequantize_per_channel(
                q_tensor,
                scale
            )

        elif granularity == "per_group":

            return dequantize_per_group(
                q_tensor,
                scale,
                group_size=params["group_size"],
                axis=params["axis"]
            )

        raise ValueError("Invalid quantization parameters")
