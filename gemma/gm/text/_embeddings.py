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

"""Utilities to extract sentence embeddings from Gemma hidden states."""

import dataclasses
from collections.abc import Sequence

from gemma.gm.nn import _config
from gemma.gm.nn import _transformer_like
from gemma.gm.text import _tokenizer
from gemma.gm.utils import _cache_helper
from gemma.gm.utils import _types
from gemma.gm.typing import _common
import jax
import jax.numpy as jnp
import numpy as np

_PADDING_ID = 0


@dataclasses.dataclass(frozen=True, kw_only=True)
class FixedPrefixEosEmbedder:
  """EOS embedder that always reuses the same prefix cache.

  The prefix is encoded once during initialization. Subsequent calls to
  :meth:`encode` reuse that frozen prefix cache and only run the suffix tokens.
  This is useful when you want a stable left context but do not want each call
  to chain into the previous one.
  """

  model: _transformer_like.TransformerLike
  params: _common.Params
  tokenizer: _tokenizer.Tokenizer
  prefix_text: str
  cache_length: int = 4096
  prefix_add_bos: bool = True
  prefix_add_eos: bool = True
  suffix_add_bos: bool = False
  suffix_add_eos: bool = True
  normalize: bool = True

  def __post_init__(self):
    prefix_input = _make_text_input(
        self.tokenizer,
        [self.prefix_text],
        add_bos=self.prefix_add_bos,
        add_eos=self.prefix_add_eos,
    )
    prefix_len = int(prefix_input.length_with_mm)
    if prefix_len >= self.cache_length:
      raise ValueError(
          'Prefix is longer than or equal to `cache_length`. '
          f'Got prefix length {prefix_len} and cache_length {self.cache_length}.'
      )

    prefix_cache = self.model.init_cache(
        batch_size=1,
        dtype=_dtype(self.params),
        cache_length=self.cache_length,
    )
    out = self.model.apply(  # pyrefly: ignore[unexpected-keyword]
        {'params': self.params},
        tokens=prefix_input.tokens_with_mm,
        positions=prefix_input.positions,
      attention_mask=_make_suffix_attention_mask(
        inputs_mask=prefix_input.inputs_mask,
        prefix_len=0,
        cache_length=self.cache_length,
      ),
        cache=prefix_cache,
        return_last_only=True,
    )
    if out.cache is None:
      raise ValueError('Prefix prefill did not return a cache.')

    object.__setattr__(self, '_prefix_cache', _cache_helper.Cache(out.cache))
    object.__setattr__(self, '_prefix_len', prefix_len)

  @property
  def prefix_len(self) -> int:
    return self._prefix_len

  def encode(self, texts: str | Sequence[str]) -> jax.Array:
    """Encode text(s) using the frozen prefix cache."""
    is_single = isinstance(texts, str)
    batch_texts = [texts] if is_single else list(texts)
    if not batch_texts:
      raise ValueError('`texts` must not be empty.')

    input_ = _make_text_input(
        self.tokenizer,
        batch_texts,
        add_bos=self.suffix_add_bos,
        add_eos=self.suffix_add_eos,
    )
    batch_size = input_.batch_size
    suffix_cache = _broadcast_cache(self._prefix_cache.cache, batch_size)
    positions = input_.positions + self._prefix_len
    attention_mask = _make_suffix_attention_mask(
        inputs_mask=input_.inputs_mask,
        prefix_len=self._prefix_len,
        cache_length=self.cache_length,
    )

    out = self.model.apply(  # pyrefly: ignore[unexpected-keyword]
        {'params': self.params},
        tokens=input_.tokens_with_mm,
        positions=positions,
        attention_mask=attention_mask,
        cache=suffix_cache,
        return_hidden_states=True,
        return_last_only=False,
    )
    hidden_states = out.hidden_states
    if hidden_states is None:
      raise ValueError('Model did not return hidden states.')

    embeddings = _gather_eos_embeddings(
        tokens=input_.tokens_with_mm,
        hidden_states=hidden_states,
        eos_id=self.tokenizer.special_tokens.EOS,
        normalize=self.normalize,
    )
    return embeddings[0] if is_single else embeddings


def encode_eos_embeddings(
    texts: str | Sequence[str],
    *,
    model: _transformer_like.TransformerLike,
    params: _common.Params,
    tokenizer: _tokenizer.Tokenizer,
    add_bos: bool = True,
    add_eos: bool = True,
    normalize: bool = True,
) -> jax.Array:
  """Encode text(s) as EOS-pooled embeddings.

  This helper runs a forward pass with ``return_hidden_states=True`` and takes
  the hidden state at EOS position as the sentence embedding.

  Args:
    texts: A single text or a batch of texts.
    model: Gemma transformer model.
    params: Model parameters.
    tokenizer: Tokenizer to use.
    add_bos: Whether to prepend BOS before encoding.
    add_eos: Whether to append EOS before encoding.
    normalize: Whether to L2-normalize embeddings.

  Returns:
    Embedding(s) with shape ``[D]`` for a single input, or ``[B, D]`` for a
    batch input.
  """
  is_single = isinstance(texts, str)
  input_ = _make_text_input(
      tokenizer,
      [texts] if isinstance(texts, str) else list(texts),
      add_bos=add_bos,
      add_eos=add_eos,
  )

  out = model.apply(  # pyrefly: ignore[unexpected-keyword]
      {'params': params},
      tokens=input_.tokens_with_mm,
      return_hidden_states=True,
      return_last_only=False,
  )
  hidden_states = out.hidden_states
  if hidden_states is None:
    raise ValueError('Model did not return hidden states.')

  embeddings = _gather_eos_embeddings(
      tokens=input_.tokens_with_mm,
      hidden_states=hidden_states,
      eos_id=tokenizer.special_tokens.EOS,
      normalize=normalize,
  )
  return embeddings[0] if is_single else embeddings


def _make_text_input(
    tokenizer: _tokenizer.Tokenizer,
    texts: Sequence[str],
    *,
    add_bos: bool,
    add_eos: bool,
) -> _types.Input:
  if not texts:
    raise ValueError('`texts` must not be empty.')

  tokenized = [
      tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
      for text in texts
  ]

  max_len = max(len(ids) for ids in tokenized)
  if max_len == 0:
    raise ValueError('At least one token is required for each input text.')

  tokens = np.full((len(tokenized), max_len), _PADDING_ID, dtype=np.int32)
  for i, ids in enumerate(tokenized):
    tokens[i, : len(ids)] = ids
  tokens = jnp.asarray(tokens)

  return _types.Input(
      text=tokens,
      images=None,
      config=_types.InputConfig(
          support_images=False,
          num_tokens_per_image=0,
          special_tokens=tokenizer.special_tokens,
      ),
  )


def _gather_eos_embeddings(
    *,
    tokens: jax.Array,
    hidden_states: jax.Array,
    eos_id: int,
    normalize: bool,
) -> jax.Array:
  seq_len = tokens.shape[-1]
  positions = jnp.arange(seq_len)[None, :]
  eos_pos = jnp.max(jnp.where(tokens == eos_id, positions, -1), axis=-1)
  last_non_pad = jnp.sum(tokens != _PADDING_ID, axis=-1) - 1
  gather_pos = jnp.where(eos_pos >= 0, eos_pos, last_non_pad)

  embeddings = hidden_states[jnp.arange(hidden_states.shape[0]), gather_pos]
  if normalize:
    norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / jnp.clip(norms, min=1e-12)
  return embeddings


def _broadcast_cache(
    cache: _config.Cache, batch_size: int
) -> _config.Cache:
  if batch_size == 1:
    return cache
  return jax.tree.map(
      lambda x: jnp.repeat(x, batch_size, axis=0),
      cache,
  )


def _make_suffix_attention_mask(
    *,
    inputs_mask: jax.Array,
    prefix_len: int,
    cache_length: int,
) -> jax.Array:
  seq_len = inputs_mask.shape[-1]
  positions = jnp.arange(cache_length)[None, None, :]
  causal_limit = prefix_len + jnp.arange(seq_len)[None, :, None] + 1
  mask = positions < causal_limit
  return inputs_mask[..., None] & mask


def _dtype(params: _common.Params) -> jnp.dtype:
  return jax.tree.leaves(params)[0].dtype
