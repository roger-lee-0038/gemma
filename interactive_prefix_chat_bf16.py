# Common imports
import os
from pathlib import Path

import dialog
import jax
import jax.numpy as jnp
from kauldron import kd
from orbax import checkpoint as ocp

# Gemma imports
from gemma import gm
from gemma.gm.ckpts import _checkpoint as _gemma_ckpt_internal
from gemma.gm.text import _prefill


# Point Gemma tokenizer cache to local directory so no network access is needed.
os.environ["GEMMA_CACHE_DIR"] = os.path.expanduser("~/.gemma")

# Reduce JAX memory pressure to avoid process being killed by OOM.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# Set USE_INT4=1 to enable int4 inference (saves ~half VRAM during inference).
# Set USE_INT4=0 to use full bfloat16 (default, higher quality).
USE_INT4 = os.environ.get("USE_INT4", "0") == "1"

# Optional runtime overrides.
CKPT_PATH = os.environ.get(
    "GEMMA_E4B_CKPT_PATH", "/home/roger/Downloads/gemma-4-e4b-it-flax"
)
TOKENIZER_PATH = os.environ.get(
    "GEMMA_E4B_TOKENIZER_PATH",
    "/home/roger/Downloads/gemma-4-tokenizer/tokenizer_gemma4.model",
)
MAX_NEW_TOKENS = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "256"))


def _load_gemma_params_bf16(path, *, text_only, restore_concurrent_gb):
  """Restores Gemma params directly as bfloat16.

  `gm.ckpts.load_params(params=None, ...)` restores using the checkpoint's
  on-disk dtype, which for this checkpoint is float32 (~30GB text-only).
  Passing a bfloat16 `jax.ShapeDtypeStruct` template as `params=` makes orbax
  cast each leaf to bfloat16 right after reading it (before device_put), so
  the full float32 tree is never materialized and the restored params end up
  at ~half the host RAM / GPU memory (~15GB).
  """
  ckpt = ocp.StandardCheckpointer(restore_concurrent_gb=restore_concurrent_gb)
  metadata, resolved_path = _gemma_ckpt_internal._get_metadata_and_path(
      ckpt, path)
  metadata = _gemma_ckpt_internal._CheckpointTree.shape_dtype_struct_like(
      tree=metadata)
  fp32_template = metadata.as_nested(
      remove_mm=text_only and metadata.has_mm_params).tree
  bf16_template = jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(
          shape=x.shape, dtype=jnp.bfloat16, sharding=kd.sharding.REPLICATED),
      fp32_template,
  )
  return gm.ckpts.load_params(
      resolved_path,
      params=bf16_template,
      donate=True,
      text_only=text_only,
      restore_concurrent_gb=restore_concurrent_gb,
  )


def _load_sampler() -> gm.text.Sampler:
    print("Loading model and checkpoint...")
    base_model = gm.nn.Gemma4_E4B(text_only=True)
    model = gm.nn.IntWrapper(model=base_model) if USE_INT4 else base_model

    if Path(CKPT_PATH).exists() and not Path(CKPT_PATH, "_METADATA").exists():
        raise FileNotFoundError(
            f"{CKPT_PATH} is not a Gemma JAX/Orbax checkpoint (missing _METADATA). "
            "Download the Flax variant (google/gemma-4/flax/gemma4-e4b-it), or use "
            "gm.ckpts.CheckpointPath.GEMMA4_E4B_IT."
        )
    if not Path(TOKENIZER_PATH).exists():
        raise FileNotFoundError(
            f"Tokenizer model not found: {TOKENIZER_PATH}. "
            "Set GEMMA_E4B_TOKENIZER_PATH to your tokenizer_gemma4.model path."
        )

    params = _load_gemma_params_bf16(
        CKPT_PATH,
        text_only=True,
        restore_concurrent_gb=12,
    )

    sampler = gm.text.Sampler(
        model=model,
        params=params,
        tokenizer=gm.text.Gemma4Tokenizer(path=TOKENIZER_PATH),
        cache_length=4096,
        max_out_length=max(256, MAX_NEW_TOKENS),
        pad_length=(256, 512, 1024),
    )

    print("Model loaded successfully!")
    print(f"Using {'int4' if USE_INT4 else 'bfloat16'} inference (bf16-target restore)")
    return sampler


def _build_prefix_state(sampler: gm.text.Sampler, prefix_prompt: str):
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


def main():
    sampler = _load_sampler()

    print("=" * 60)
    print("Fixed Prefix Interactive Chat (bf16-target restore)")
    print("Each turn reuses only the first prefix state (multi_turn=False style).")
    print("Type 'exit' or 'quit' to end.")
    print("=" * 60)

    while True:
        prefix_prompt = input("\nPrefix prompt: ").strip()
        if prefix_prompt:
            break
        print("Prefix prompt cannot be empty.")

    prefix_state = _build_prefix_state(sampler, prefix_prompt)
    print("Prefix state initialized. Start chatting.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            query_conv = dialog.Conversation(dialog.User(user_input))
            out = sampler.sample(
                query_conv,
                last_state=prefix_state,
                return_state=True,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            print(f"Gemma: {out.text}")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
