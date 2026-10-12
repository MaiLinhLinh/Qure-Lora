import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiAdapterLinear(nn.Linear):
    """Frozen Linear plus configurable low-rank adapter deltas.

    The base weight is always frozen. ``separate`` keeps independent
    reference/candidate LoRA parameters; ``shared`` aliases both adapter
    names to one parameter set. ``base`` is an explicit bypass used by the
    query-only and candidate-only ablations.
    """

    ADAPTER_NAMES = ("reference", "candidate")
    VALID_MODES = ("separate", "shared")
    VALID_ACTIVE_ADAPTERS = ("reference", "candidate", "base")

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=8,
        alpha=16,
        dropout=0.05,
        adapter_names=ADAPTER_NAMES,
        lora_mode="separate",
        lora_lambda=1.0,
        device=None,
        dtype=None,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if lora_mode not in self.VALID_MODES:
            raise ValueError(
                f"unknown lora_mode {lora_mode!r}; "
                f"expected one of {self.VALID_MODES}"
            )
        if lora_lambda < 0:
            raise ValueError("lora_lambda must be non-negative")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)

        adapter_names = tuple(adapter_names)
        if adapter_names != self.ADAPTER_NAMES:
            raise ValueError(
                "MultiAdapterLinear requires exactly the adapters "
                "reference and candidate"
            )

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = float(dropout)
        self.lora_mode = lora_mode
        self.lora_lambda = float(lora_lambda)
        self.merge_weights = False
        self.active_adapter = "candidate"

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        if lora_mode == "shared":
            shared_A = nn.Parameter(torch.empty(self.rank, in_features, **factory_kwargs))
            shared_B = nn.Parameter(torch.empty(out_features, self.rank, **factory_kwargs))
            self.lora_A["reference"] = shared_A
            self.lora_A["candidate"] = shared_A
            self.lora_B["reference"] = shared_B
            self.lora_B["candidate"] = shared_B
            self._reset_lora_parameters("reference")
        else:
            for adapter_name in self.ADAPTER_NAMES:
                self.lora_A[adapter_name] = nn.Parameter(
                    torch.empty(self.rank, in_features, **factory_kwargs)
                )
                self.lora_B[adapter_name] = nn.Parameter(
                    torch.empty(out_features, self.rank, **factory_kwargs)
                )
                self._reset_lora_parameters(adapter_name)

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def _reset_lora_parameters(self, adapter_name):
        nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[adapter_name])

    @classmethod
    def from_linear(
        cls,
        linear,
        rank=8,
        alpha=16,
        dropout=0.05,
        lora_mode="separate",
        lora_lambda=1.0,
    ):
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            lora_mode=lora_mode,
            lora_lambda=lora_lambda,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            wrapped.weight.copy_(linear.weight)
            if linear.bias is not None:
                wrapped.bias.copy_(linear.bias)
        return wrapped

    def set_active_adapter(self, adapter_name):
        if adapter_name not in self.VALID_ACTIVE_ADAPTERS:
            raise ValueError(
                f"unknown adapter {adapter_name!r}; "
                f"expected one of {self.VALID_ACTIVE_ADAPTERS}"
            )
        self.active_adapter = adapter_name

    def _parameters_for_active_adapter(self):
        if self.active_adapter == "base":
            return None, None
        return (
            self.lora_A[self.active_adapter],
            self.lora_B[self.active_adapter],
        )

    def forward(self, x):
        base_output = F.linear(x, self.weight, self.bias)
        A, B = self._parameters_for_active_adapter()
        if A is None:
            return base_output

        delta = F.dropout(x, p=self.lora_dropout, training=self.training)
        delta = delta @ A.transpose(-1, -2)
        delta = delta @ B.transpose(-1, -2)
        delta = delta * self.scaling * self.lora_lambda
        return base_output + delta
