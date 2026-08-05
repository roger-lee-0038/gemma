# Common imports
import os
from pathlib import Path

# import certifi

# Gemma imports
from gemma import gm

# Force Python/TF/libcurl to use certifi CA bundle instead of missing
# /etc/ssl/certs/ca-certificates.crt on this machine.
# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
# os.environ["CURL_CA_BUNDLE"] = certifi.where()

# Point Gemma tokenizer cache to local directory so no network access is needed.
os.environ["GEMMA_CACHE_DIR"] = os.path.expanduser("~/.gemma")

# Reduce JAX memory pressure to avoid process being killed by OOM.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Loading weights (bfloat16) needs ~10.5GB per shard; IntWrapper quantizes at
# inference time only. Use 0.9 to give enough headroom for the load peak.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# Set USE_INT4=1 to enable int4 inference (saves ~half VRAM during inference).
# Set USE_INT4=0 to use full bfloat16 (default, higher quality).
USE_INT4 = os.environ.get("USE_INT4", "0") == "1"

print("Loading model and checkpoint...")
base_model = gm.nn.Gemma4_E4B(text_only=True)
# Int4 wrapper replaces Dense/Einsum layers with int4 ops at runtime.
# Weights are still loaded in bfloat16 from the checkpoint, then quantized on
# the fly during inference only.
model = gm.nn.IntWrapper(model=base_model) if USE_INT4 else base_model

# If you provide a local path, it must be an Orbax/Flax checkpoint directory
# containing _METADATA (not a Transformers safetensors export).
ckpt_path = "/home/roger/Downloads/gemma-4-e4b-it-flax"
if Path(ckpt_path).exists() and not Path(ckpt_path, "_METADATA").exists():
    raise FileNotFoundError(
        f"{ckpt_path} is not a Gemma JAX/Orbax checkpoint (missing _METADATA). "
        "Download the Flax variant (google/gemma-4/flax/gemma4-e4b-it), or use "
        "gm.ckpts.CheckpointPath.GEMMA4_E4B_IT."
    )

params = gm.ckpts.load_params(
    ckpt_path,
    text_only=True,
    restore_concurrent_gb=12,
)

sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
    print_stream=False,
    tokenizer=gm.text.Gemma4Tokenizer(path="/home/roger/Downloads/gemma-4-tokenizer/tokenizer_gemma4.model")
)

print("Model loaded successfully!")
print(f"Using {'int4' if USE_INT4 else 'bfloat16'} inference")
print("=" * 60)
print("Interactive Chat (type 'exit' or 'quit' to end)")
print("=" * 60)

# Interactive loop
while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        response = sampler.chat(user_input)
        print(f"Gemma: {response}")
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue
