from typing import Any, Optional, Sequence
import torch

class SliceUpdateKeyValueCache:
    def __init__(
        self,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.k = torch.zeros(shape, dtype=dtype) if k is None else k
        self.v = torch.zeros(shape, dtype=dtype) if v is None else v