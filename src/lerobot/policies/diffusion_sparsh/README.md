# Diffusion-Sparsh for LeRobot

This package is a `diffusion_baseline`-derived LeRobot policy renamed to `diffusion_sparsh`.

The main change is the tactile raw branch:

```text
observation.images.tac_raw.<sensor>
    -> temporal pair I_t + I_{t-k}
    -> official Sparsh ViT encoder from facebookresearch/sparsh
    -> projection head
    -> global_cond
    -> DiT / UNet diffusion denoiser
```

The original ResNet tactile raw encoder is still available by setting:

```python
tactile_raw_encoder_type = "resnet"
```

## Files

Place the files like this:

```bash
mkdir -p lerobot/policies/diffusion_sparsh
cp configuration_diffusion_sparsh.py lerobot/policies/diffusion_sparsh/configuration_diffusion_sparsh.py
cp modeling_diffusion_sparsh.py lerobot/policies/diffusion_sparsh/modeling_diffusion_sparsh.py
cp processor_diffusion_sparsh.py lerobot/policies/diffusion_sparsh/processor_diffusion_sparsh.py
```

Then make sure your LeRobot policy import/registry path imports this folder in the same way as your other custom policies.

## Dependencies

Install the official Sparsh repo inside the same environment as LeRobot:

```bash
git clone https://github.com/facebookresearch/sparsh.git ~/third_party/sparsh
pip install -e ~/third_party/sparsh
pip install huggingface_hub safetensors
```

If you do not install the repo editable, set:

```python
sparsh_repo_path = "~/third_party/sparsh"
```

in `DiffusionSparshConfig` so the model can import `tactile_ssl.model`.

## Default Sparsh settings

The default config uses the official DINO base Sparsh checkpoint:

```python
tactile_raw_encoder_type = "sparsh"
sparsh_model_name = "facebook/sparsh-dino-base"
sparsh_checkpoint_filename = "dino_vitbase.ckpt"
sparsh_ssl_name = "dino"
sparsh_model_size = "base"
sparsh_input_channels = 6
sparsh_input_size = (320, 240)
sparsh_num_register_tokens = 1
sparsh_pos_embed_fn = "sinusoidal"
sparsh_pooling = "mean_patch"
sparsh_frozen = True
sparsh_projection_dim = 128
sparsh_temporal_stride = 1
```

These settings mirror the official Sparsh DINO ViT configuration: `vit_base`, 6-channel tactile pair input, sinusoidal positional embeddings, and one register token.

## Temporal pairing

For each observation step, the encoder builds:

```text
pair_i = concat(I_i, I_{max(0, i-k)})
```

where `k = sparsh_temporal_stride`.

For your current default `n_obs_steps=2`, use:

```python
sparsh_temporal_stride = 1
```

If you later set `n_obs_steps >= 6`, you can use:

```python
sparsh_temporal_stride = 5
```

which is closer to the original Sparsh temporal pairing.

## Important notes

1. The official Sparsh ViT returns patch tokens `(B, N, D)`, not a single CLS vector. This wrapper uses mean patch pooling by default.
2. The Sparsh backbone is frozen by default. Only the projection head and the diffusion policy are trained.
3. The policy still supports `tac_depth + tac_normal` and marker displacement exactly as before.
4. No MoE is enabled by default. Start with single DiT denoiser first.

## Smoke test suggestion

Before long training, run one forward/backward batch and check that you see a log like:

```text
[Sparsh] official builder loaded ... tensors from facebook/sparsh-dino-base/dino_vitbase.ckpt
```

If the loaded ratio is low, check:

```python
sparsh_model_size
sparsh_num_register_tokens
sparsh_input_size
sparsh_checkpoint_filename
sparsh_ssl_name
```

The most common mismatch is using a checkpoint/config pair that does not match the official Sparsh model size or register-token setting.
