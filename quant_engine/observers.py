
import torch

from .utils import get_quantization_range


class MinMaxObserver:

    def __init__(self, bits=8):

        q_min, q_max = get_quantization_range(bits)

        self.bits = bits
        self.q_min = q_min
        self.q_max = q_max

        self.min_val = None
        self.max_val = None

        self.num_batches = 0
        self.num_values = 0

    def update(self, tensor):

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if tensor.numel() == 0:
            raise ValueError("Cannot observe an empty tensor")

        if not torch.isfinite(tensor).all():
            raise ValueError("Tensor contains NaN or Inf")

        current_min = tensor.detach().amin().item()
        current_max = tensor.detach().amax().item()

        if self.min_val is None:

            self.min_val = current_min
            self.max_val = current_max

        else:

            self.min_val = min(
                self.min_val,
                current_min
            )

            self.max_val = max(
                self.max_val,
                current_max
            )

        self.num_batches += 1
        self.num_values += tensor.numel()

    def calculate_qparams(self):

        if self.min_val is None:
            raise RuntimeError(
                "Observer has not collected any data"
            )

        max_abs = max(
            abs(self.min_val),
            abs(self.max_val)
        )

        # Symmetric quantization
        scale = max_abs / self.q_max if max_abs != 0 else 1.0

        scale = torch.tensor(
            scale,
            dtype=torch.float32
        )

        zero_point = 0

        return scale, zero_point

    def reset(self):

        self.min_val = None
        self.max_val = None

        self.num_batches = 0
        self.num_values = 0

    def get_statistics(self):

        if self.min_val is None:
            raise RuntimeError(
                "Observer has not collected any data"
            )

        return {
            "min": self.min_val,
            "max": self.max_val,
            "num_batches": self.num_batches,
            "num_values": self.num_values
        }


class ModelCalibrator:

    def __init__(self, bits=8,observer_type="minmax",percentile=99.9):

        get_quantization_range(bits)

        if observer_type not in ["minmax", "percentile"]:
                    raise ValueError("Invalid observer type")

        self.bits = bits
        self.observer_type = observer_type
        self.percentile = percentile
        self.observers = {}

    def _create_observer(self):

        if self.observer_type == "minmax":

            return MinMaxObserver(
                bits=self.bits
            )

        return PercentileObserver(
            bits=self.bits,
            percentile=self.percentile
        )

    def calibrate(self, model, calibration_data):

        self.observers = {}

        handles = []

        # Preserve individual training states
        training_states = {
            module: module.training
            for module in model.modules()
        }

        def create_hook(name):

            def hook(module, inputs, output):

                if not isinstance(output, torch.Tensor):
                    return

                if name not in self.observers:

                    self.observers[name] = self._create_observer()

                self.observers[name].update(output)

            return hook

        try:

            model.eval()

            # Attach observers to leaf modules
            for name, module in model.named_modules():

                if not name:
                    continue

                if len(list(module.children())) == 0:

                    handle = module.register_forward_hook(
                        create_hook(name)
                    )

                    handles.append(handle)

            # Run calibration
            with torch.inference_mode():

                for batch in calibration_data:

                    # Support datasets returning (input, label)
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    model(inputs)

        finally:

            # Always remove hooks
            for handle in handles:
                handle.remove()

            # Restore original training states
            for module, training in training_states.items():
                module.training = training

        if not self.observers:
            raise RuntimeError(
                "No tensor outputs were observed during calibration"
            )

        return self.observers

    def get_calibration_results(self):

        results = {}

        for name, observer in self.observers.items():

            scale, zero_point = observer.calculate_qparams()

            results[name] = {
                **observer.get_statistics(),
                "scale": scale,
                "zero_point": zero_point,
                "bits": self.bits
            }

        return results


class PercentileObserver(MinMaxObserver):

    def __init__(
        self,
        bits=8,
        percentile=99.9,
        max_samples=200000,
        samples_per_update=20000
    ):

        super().__init__(bits=bits)

        if not isinstance(percentile, (int, float)):
            raise TypeError("percentile must be numeric")

        if not 0 < percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")

        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be positive")

        if not isinstance(samples_per_update, int) or samples_per_update <= 0:
            raise ValueError("samples_per_update must be positive")

        self.percentile = percentile
        self.max_samples = max_samples
        self.samples_per_update = samples_per_update

        self.samples = torch.empty(0, dtype=torch.float32)
        self.clipping_threshold = None

    def update(self, tensor):

        # Maintain global min/max and observation counts
        super().update(tensor)

        values = tensor.detach().abs().reshape(-1)

        # Sample large activation tensors to limit memory usage
        if values.numel() > self.samples_per_update:

            indices = torch.randperm(
                values.numel(),
                device=values.device
            )[:self.samples_per_update]

            values = values[indices]

        values = values.float().cpu()

        combined = torch.cat([
            self.samples,
            values
        ])

        # Keep a bounded random sample
        if combined.numel() > self.max_samples:

            indices = torch.randperm(
                combined.numel()
            )[:self.max_samples]

            combined = combined[indices]

        self.samples = combined

    def calculate_qparams(self):

        if self.samples.numel() == 0:
            raise RuntimeError(
                "Observer has not collected any data"
            )

        threshold = torch.quantile(
            self.samples,
            self.percentile / 100.0
        )

        self.clipping_threshold = threshold.item()

        scale = (
            threshold / self.q_max
            if threshold.item() != 0
            else torch.tensor(1.0)
        )

        return scale.float(), 0

    def get_statistics(self):

        stats = super().get_statistics()

        stats.update({
            "percentile": self.percentile,
            "clipping_threshold": self.clipping_threshold,
            "sample_count": self.samples.numel()
        })

        return stats

    def reset(self):

        super().reset()

        self.samples = torch.empty(
            0,
            dtype=torch.float32
        )

        self.clipping_threshold = None
