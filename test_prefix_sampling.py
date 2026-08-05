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

"""Integration test for fixed-prefix sampling without multi-turn chaining."""

from __future__ import annotations

import os
from pathlib import Path

import dialog
import jax
import pytest

from gemma import gm
from gemma.gm.text import _prefill


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


def _load_sampler() -> gm.text.Sampler:
  _skip_if_missing_resources()

  model = gm.nn.Gemma4_E4B(text_only=True)
  params = gm.ckpts.load_params(
      _get_ckpt_path(),
      text_only=True,
      restore_concurrent_gb=12,
  )
  tokenizer = gm.text.Gemma4Tokenizer(path=_get_tokenizer_path())

  return gm.text.Sampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      cache_length=4096,
      max_out_length=256,
      pad_length=(256, 512, 1024),
  )


def _build_prefix_state(
    sampler: gm.text.Sampler,
    prefix_prompt: str,
):
  prefix_conv = dialog.Conversation(dialog.User(prefix_prompt))
  inputs = sampler._get_inputs(  # pylint: disable=protected-access
      prompt=prefix_conv,
      images=None,
      add_bos=True,
      has_batch_dim=False,
      sharding=None,
  )
  return _prefill.prefill(
      model=sampler.model,
      params=sampler.params,
      input=inputs,
      last_state=None,
      cache_length=sampler.cache_length,
      max_out_length=sampler.max_out_length,
      pad_length=sampler.pad_length,
      rng=jax.random.PRNGKey(0),
      sharding=None,
  )


def test_fixed_prefix_sampling_reuses_only_first_prefix_state():
  sampler = _load_sampler()

  prefix_state = _build_prefix_state(
      sampler,
      "You are a concise assistant. Reply in one sentence.",
  )
  base_used_cache_length = int(prefix_state.used_cache_length)

  q1 = dialog.Conversation(dialog.User("What is 1 + 3?"))
  q2 = dialog.Conversation(dialog.User("Write a short poem about lakes."))

  out1 = sampler.sample(
      q1,
      last_state=prefix_state,
      return_state=True,
      max_new_tokens=64,
  )
  out1_again = sampler.sample(
      q1,
      last_state=prefix_state,
      return_state=True,
      max_new_tokens=64,
  )

  # Same fixed prefix + same query should be stable under greedy decoding.
  assert out1.text == out1_again.text

  # The reusable prefix state itself should remain unchanged across calls.
  assert int(prefix_state.used_cache_length) == base_used_cache_length

  fixed_q2 = sampler.sample(
      q2,
      last_state=prefix_state,
      return_state=True,
      max_new_tokens=64,
  )
  chained_q2 = sampler.sample(
      q2,
      last_state=out1.state,
      return_state=True,
      max_new_tokens=64,
  )

  # Chained state should have consumed more cache than fixed-prefix mode.
  assert int(chained_q2.state.used_cache_length) > int(
      fixed_q2.state.used_cache_length
  )


if __name__ == "__main__":
  test_fixed_prefix_sampling_reuses_only_first_prefix_state()
