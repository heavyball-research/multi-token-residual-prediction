# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# Opt-in deterministic mode via env var. Must run BEFORE any torch CUDA op so
# CUBLAS_WORKSPACE_CONFIG takes effect; we also force this here in Python
# (even though the shell script exports it) as a belt-and-suspenders measure.
# Enabled when DETERMINISTIC=1/true/yes. Disabled (default) is fastest.
_det_raw = os.environ.get("DETERMINISTIC", "").strip().lower()
_DETERMINISTIC = _det_raw in ("1", "true", "yes")
if _DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

if _DETERMINISTIC:
    # warn_only=True: if an op lacks a deterministic kernel, fall back instead
    # of raising — preferable for training where a hard error would kill the
    # run on the first non-deterministic op we haven't audited.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(
        f"[launcher] DETERMINISTIC=True: use_deterministic_algorithms=True(warn_only), "
        f"cudnn.deterministic=True, cudnn.benchmark=False, "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
        flush=True,
    )

from llamafactory.train.tuner import run_exp  # noqa: E402  use absolute import


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
