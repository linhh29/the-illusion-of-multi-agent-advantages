# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Delegate to verl so the patch runs before tensordict in all processes (including Ray workers).
from __future__ import annotations

import verl._torch_tensordict_compat  # noqa: F401
