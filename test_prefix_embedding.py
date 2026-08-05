# Copyright 2026 DeepMind Technologies Limited.
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

"""Integration test for fixed-prefix Gemma EOS embeddings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import numpy as np

from gemma import gm
from gemma.gm.text import _embeddings


os.environ["GEMMA_CACHE_DIR"] = os.path.expanduser("~/.gemma")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


def _get_ckpt_path() -> str:
  return os.environ.get(
      "GEMMA_E4B_CKPT_PATH",
      "/home/roger/Downloads/gemma-4-e4b-it-flax",
  )


def _get_tokenizer_path() -> str:
  return os.environ.get(
      "GEMMA_E4B_TOKENIZER_PATH",
      "/home/roger/Downloads/gemma-4-tokenizer/tokenizer_gemma4.model",
  )


def _skip_if_missing_resources() -> None:
  ckpt_path = Path(_get_ckpt_path())
  tokenizer_path = Path(_get_tokenizer_path())

  if not ckpt_path.exists():
    pytest.skip(f"Missing checkpoint path: {ckpt_path}")
  if not tokenizer_path.exists():
    pytest.skip(f"Missing tokenizer path: {tokenizer_path}")
  if not Path(ckpt_path, "_METADATA").exists():
    pytest.skip(
        f"{ckpt_path} is not a Gemma JAX/Orbax checkpoint (missing _METADATA)"
    )


def _load_model_and_params():
  _skip_if_missing_resources()

  base_model = gm.nn.Gemma4_E4B(text_only=True)
  params = gm.ckpts.load_params(
      _get_ckpt_path(),
      text_only=True,
      restore_concurrent_gb=12,
  )
  tokenizer = gm.text.Gemma4Tokenizer(path=_get_tokenizer_path())
  return base_model, params, tokenizer


def test_fixed_prefix_embeddings_are_stable_and_distinguish_texts():
  model, params, tokenizer = _load_model_and_params()

  embedder = _embeddings.FixedPrefixEosEmbedder(
      model=model,
      params=params,
      tokenizer=tokenizer,
      prefix_text="You are a concise assistant that answers with one sentence.",
      cache_length=256,
  )

  cat_like = [
      "The cat sits on the mat.",
      "A cat is sitting on a mat.",
  ]
  unrelated = [
      "Quantum mechanics describes particles and waves.",
  ]

  first = embedder.encode(cat_like[0])
  second = embedder.encode(cat_like[1])
  first_again = embedder.encode(cat_like[0])

  assert first.shape == first_again.shape
  assert first.ndim == 1
  np.testing.assert_allclose(first, first_again, rtol=1e-5, atol=1e-5)

  batch = embedder.encode(cat_like + unrelated)
  similarity = batch @ batch.T
  print("Similarity matrix:\n", similarity, flush=True)

  assert similarity.shape == (3, 3)
  assert similarity[0, 0] == pytest.approx(1.0, abs=1e-4)
  assert similarity[1, 1] == pytest.approx(1.0, abs=1e-4)
  assert similarity[2, 2] == pytest.approx(1.0, abs=1e-4)
  assert similarity[0, 1] > similarity[0, 2]
  assert similarity[1, 0] > similarity[1, 2]
  assert not np.isclose(similarity[0, 1], similarity[0, 2])


if __name__ == "__main__":
  test_fixed_prefix_embeddings_are_stable_and_distinguish_texts()
