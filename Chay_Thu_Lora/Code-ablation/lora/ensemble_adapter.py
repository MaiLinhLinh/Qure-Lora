"""LoRA factors for a parameter-efficient ensemble.

This module is intentionally separate from ``multi_adapter.py``.  The
existing two-role adapter implementation is used by Exp 0--7 and keeps its
original state-dict layout.  Exp 9 uses this class, where every ensemble
member owns an independent reference/candidate pair while the base Linear
weight is shared and frozen.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class EnsembleMultiAdapterLinear(nn.Linear):
    """Frozen Linear plus independent LoRA factors for each member and role.

    A member is a complete collection of LoRA factors across all injected
    layers.  Each layer therefore stores two factors for every member:
    ``reference`` and ``candidate``.  The base weight and bias are shared by
    all members and are never trainable.
    """

    ADAPTER_NAMES = ("reference", "candidate")
    VALID_ACTIVE_ADAPTERS = ("reference", "candidate", "base")

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=8,
        alpha=16,
        dropout=0.05,
        num_members=4,
        lora_lambda=1.0,
        device=None,
        dtype=None,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if int(num_members) <= 1:
            raise ValueError("num_members must be greater than one")
        if lora_lambda < 0:
            raise ValueError("lora_lambda must be non-negative")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = float(dropout)
        self.num_members = int(num_members)
        self.lora_lambda = float(lora_lambda)
        self.merge_weights = False
        self.active_member = 0
        self.active_adapter = "candidate"

        # ParameterDict gives stable, explicit state-dict names such as
        # lora_A.member_0_reference.  The underscore is deliberate: dots are
        # reserved by PyTorch for module hierarchy separators.
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for member_idx in range(self.num_members):
            for adapter_name in self.ADAPTER_NAMES:
                key = self._key(member_idx, adapter_name)
                self.lora_A[key] = nn.Parameter(
                    torch.empty(self.rank, in_features, **factory_kwargs)
                )
                self.lora_B[key] = nn.Parameter(
                    torch.empty(out_features, self.rank, **factory_kwargs)
                )
                self._reset_lora_parameters(key)

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @staticmethod
    def _key(member_idx, adapter_name):
        return f"member_{int(member_idx)}_{adapter_name}"

    def _reset_lora_parameters(self, key):
        # The random A matrices provide member-specific directions.  B starts
        # at zero so every member is identity-preserving at initialization.
        nn.init.kaiming_uniform_(self.lora_A[key], a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[key])

    @classmethod
    def from_linear(
        cls,
        linear,
        rank=8,
        alpha=16,
        dropout=0.05,
        num_members=4,
        lora_lambda=1.0,
    ):
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_members=num_members,
            lora_lambda=lora_lambda,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            wrapped.weight.copy_(linear.weight)
            if linear.bias is not None:
                wrapped.bias.copy_(linear.bias)
        return wrapped

    def set_active_member(self, member_idx):
        member_idx = int(member_idx)
        if not 0 <= member_idx < self.num_members:
            raise ValueError(
                f"unknown ensemble member {member_idx}; "
                f"expected an index in [0, {self.num_members - 1}]"
            )
        self.active_member = member_idx

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
        key = self._key(self.active_member, self.active_adapter)
        return self.lora_A[key], self.lora_B[key]

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
