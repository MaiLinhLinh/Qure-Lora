import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiAdapterLinear(nn.Linear):
    """A frozen linear base with separate low-rank deltas per adapter."""

    ADAPTER_NAMES = ("reference", "candidate")

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=8,
        alpha=16,
        dropout=0.05,
        adapter_names=ADAPTER_NAMES,
        device=None,
        dtype=None,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

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
        self.merge_weights = False
        self.active_adapter = "candidate"

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
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
    def from_linear(cls, linear, rank=8, alpha=16, dropout=0.05):
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            wrapped.weight.copy_(linear.weight)
            if linear.bias is not None:
                wrapped.bias.copy_(linear.bias)
        return wrapped

    def set_active_adapter(self, adapter_name):
        if adapter_name not in self.ADAPTER_NAMES:
            raise ValueError(
                f"unknown adapter {adapter_name!r}; "
                f"expected one of {self.ADAPTER_NAMES}"
            )
        self.active_adapter = adapter_name

    def forward(self, x):
        base_output = F.linear(x, self.weight, self.bias)

        A = self.lora_A[self.active_adapter]
        B = self.lora_B[self.active_adapter]

        delta = F.dropout(x, p=self.lora_dropout, training=self.training)
        delta = delta @ A.transpose(-1, -2)
        delta = delta @ B.transpose(-1, -2)
        delta = delta * self.scaling

        return base_output + delta
