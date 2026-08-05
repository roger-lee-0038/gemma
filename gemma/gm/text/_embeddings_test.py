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

from gemma import gm
from gemma.gm.text import _embeddings
import jax
import jax.numpy as jnp
import numpy as np


def _make_model_and_params():
  model = gm.testing.DummyGemma()
  variables = model.init(
      jax.random.PRNGKey(0),
      jnp.zeros((5,), dtype=jnp.int32),
  )
  params = variables['params']
  tokenizer = gm.testing.DummyTokenizer()
  return model, params, tokenizer


def test_encode_eos_embeddings_batch_shape_and_norm():
  model, params, tokenizer = _make_model_and_params()

  embeddings = _embeddings.encode_eos_embeddings(
      ['hello world', 'Hello there !'],
      model=model,  # pyrefly: ignore[bad-argument-type]
      params=params,
      tokenizer=tokenizer,
  )

  assert embeddings.ndim == 2
  assert embeddings.shape[0] == 2
  np.testing.assert_allclose(
      jnp.linalg.norm(embeddings, axis=-1),
      jnp.ones((2,)),
      rtol=1e-5,
      atol=1e-5,
  )


def test_encode_eos_embeddings_matches_manual_eos_gather():
  model, params, tokenizer = _make_model_and_params()
  texts = ['hello world', 'Hello there !']

  embeddings = _embeddings.encode_eos_embeddings(
      texts,
      model=model,  # pyrefly: ignore[bad-argument-type]
      params=params,
      tokenizer=tokenizer,
      add_bos=True,
      add_eos=True,
      normalize=False,
  )

  tokenized = [
      tokenizer.encode(text, add_bos=True, add_eos=True) for text in texts
  ]
  max_len = max(len(ids) for ids in tokenized)
  tokens = np.zeros((len(tokenized), max_len), dtype=np.int32)
  for i, ids in enumerate(tokenized):
    tokens[i, : len(ids)] = ids
  tokens = jnp.asarray(tokens)

  out = model.apply(  # pyrefly: ignore[unexpected-keyword]
      {'params': params},
      tokens=tokens,
      return_hidden_states=True,
      return_last_only=False,
  )
  hidden_states = out.hidden_states

  eos_id = tokenizer.special_tokens.EOS
  positions = jnp.arange(tokens.shape[-1])[None, :]
  eos_pos = jnp.max(jnp.where(tokens == eos_id, positions, -1), axis=-1)
  expected = hidden_states[jnp.arange(hidden_states.shape[0]), eos_pos]

  np.testing.assert_allclose(embeddings, expected, rtol=1e-5, atol=1e-5)


def test_encode_eos_embeddings_single_text_returns_vector():
  model, params, tokenizer = _make_model_and_params()

  embedding = _embeddings.encode_eos_embeddings(
      'hello world',
      model=model,  # pyrefly: ignore[bad-argument-type]
      params=params,
      tokenizer=tokenizer,
      normalize=False,
  )

  assert embedding.ndim == 1


def test_encode_eos_embeddings_prompt_similarity_changes():
  model, params, tokenizer = _make_model_and_params()
  prompts = ['hello world', 'hello world', 'Hello there !']

  embeddings = _embeddings.encode_eos_embeddings(
      prompts,
      model=model,  # pyrefly: ignore[bad-argument-type]
      params=params,
      tokenizer=tokenizer,
  )

  similarity = embeddings @ embeddings.T

  np.testing.assert_allclose(similarity[0, 1], 1.0, rtol=1e-5, atol=1e-5)
  assert not np.isclose(similarity[0, 2], 1.0)
  assert not np.isclose(similarity[1, 2], 1.0)
