---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Building a Choral Source Separator with SepReformer in JAX

This tutorial demonstrates how to perform audio source separation using the `SepReformer`
architecture. The original paper can be found at https://arxiv.org/abs/2406.05983; the authors' full source code is at https://github.com/dmlguq456/SepReformer

```python
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Array
import soundfile as sf
```

## The Task

Given a mono mixture waveform $x \in \mathbb{R}^T$, produce $N$ separated stem
waveforms $\hat{s}_1, \ldots, \hat{s}_N \in \mathbb{R}^T$ such that
$\sum_n \hat{s}_n \approx x$ and each $\hat{s}_n$ matches one isolated voice
track.

We use the **JaCappella** corpus (35 a cappella songs) via Hugging Face. Each
song has 5 isolated stems — `lead_vocal`, `soprano`, `alto`, `tenor`, `bass` —
and the mixture is their sum.

```python
import librosa
from pathlib import Path

STEM_NAMES = ("lead_vocal", "soprano", "alto", "tenor", "bass")
SAMPLE_RATE = 44100

class JaCappellaDataset:
    def __init__(self, root) -> None:
        self.root = Path(root)
        self.sample_rate = 44100
        self.songs = sorted(
            song
            for genre in self.root.iterdir()
            if genre.is_dir() and not genre.name.startswith(".")
            for song in genre.iterdir()
            if song.is_dir() and not song.name.startswith("."))

    def _load_wav(self, path: Path) -> Float[np.ndarray, "T"]:
        """Load a wav file, resample if needed, return mono float32."""
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # take first channel if stereo
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio

    def _load_stems(self, song_dir: Path) -> Float[np.ndarray, "N T"]:
        """Load all available stems for a song."""
        stems = []
        for i, name in enumerate(STEM_NAMES):
            path = song_dir / f"{name}.wav"
            stems.append(self._load_wav(path) if path.exists() else np.zeros(0, dtype=np.float32))
        max_len = max(len(s) for s in stems)
        result = np.zeros((5, max_len), dtype=np.float32)
        for i, s in enumerate(stems):
            result[i, :len(s)] = s
        return result

    def __getitem__(self, idx: int) -> tuple[Float[np.ndarray, "T"], Float[np.ndarray, "N T"]]:
        """Load full song."""
        song_dir = self.songs[idx]
        stems = self._load_stems(song_dir)
        return stems.sum(axis=0), stems

    def __len__(self) -> int:
        return len(self.songs)

dataset = JaCappellaDataset("data/jacappella")
```

We'll feed the dataset to our model using the `grain` library.

```python
import grain

SEG_SAMPLES = SAMPLE_RATE * 2   # 2-second segments
BATCH_SIZE  = 1

def extract_segment(item, rng):
    mixture, stems = item
    T = mixture.shape[0]
    if T <= SEG_SAMPLES:
        pad = SEG_SAMPLES - T
        return np.pad(mixture, (0, pad)), np.pad(stems, ((0, 0), (0, pad)))
    start = rng.integers(0, T - SEG_SAMPLES)
    return mixture[start:start + SEG_SAMPLES], stems[:, start:start + SEG_SAMPLES]

def batch_to_jax(items):
    mixtures = jnp.array(np.stack([m for m, _ in items]))  # (B, T)
    stems    = jnp.array(np.stack([s for _, s in items]))  # (B, N, T)
    return mixtures, stems

loader = (grain.MapDataset.source(dataset)
        .seed(0).shuffle()
        .random_map(extract_segment)
        .batch(BATCH_SIZE, drop_remainder=True, batch_fn=batch_to_jax))
```

To make sure the data is loading correctly, we can sample a batch and log it to Tensorboard.

```python
from tensorboardX import SummaryWriter

mixture, stems = next(iter(loader))
mixture = np.array(mixture[0])  # (T,)
stems = np.array(stems[0])      # (N, T)
writer = SummaryWriter("samples")
peak = np.max(np.abs(mixture))
scale = 0.99 / peak if peak > 0 else 1.0
writer.add_audio(
    f"mixture",
    mixture * scale,
    sample_rate=dataset.sample_rate)
for n in range(stems.shape[0]):
    writer.add_audio(
        f"stem/{n}",
        stems[n] * scale,
        sample_rate=dataset.sample_rate)
writer.close()
```

```python
from IPython.display import display, HTML
display(HTML(open("mixture.html").read()))
display(HTML(open("stem.html").read()))
```

## Architecture


At a high level:
- The input waveform is encoded to a latent space using convolutional and transformer layers.
- The result gets split into separate pieces for each voice part
- Each piece is decoded back to a waveform by the same stack of transformer and convolutional layers.

$$x \xrightarrow{\text{Conv}} h \xrightarrow{\text{Enc blocks}} h \xrightarrow{\text{Split}} \{h_n\} \xrightarrow{\text{Dec blocks}} \xrightarrow{\text{ConvT}} \{\hat{s}_n\}$$

## Convolutional Layers: Waveform → Latent Frames and Back

A strided 1-D convolution converts the raw waveform into a sequence of latent
frames. With kernel $K$ and stride $S$:

$$L = \left\lfloor \frac{T - K}{S} \right\rfloor + 1$$

The encoder applies a GELU after the convolution:

$$h = \text{GELU}(W_\text{enc} * x), \quad h \in \mathbb{R}^{L \times C}$$

```python
class Encoder(nnx.Module):
    def __init__(self, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(1, out_channels, kernel_size=(16,), strides=(8,),
                             padding='VALID', rngs=rngs)

    def __call__(self, x: Float[Array, "B T"]) -> Float[Array, "B L C"]:
        h = self.conv(x[..., None])   # (B, T, 1) -> (B, L, C)
        return jax.nn.gelu(h)
```

```python
class Decoder(nnx.Module):
    def __init__(self, in_channels: int, *, rngs: nnx.Rngs):
        self.conv_t = nnx.ConvTranspose(in_channels, 1, kernel_size=(16,), strides=(8,),
                                        padding='VALID', rngs=rngs)

    def __call__(self, h: Float[Array, "B L C"]) -> Float[Array, "B T"]:
        out = self.conv_t(h)    # (B, L, C) -> (B, T, 1)
        return out[..., 0]      # (B, T)
```

Default: $K=16$, $S=8$, $C=256$. At 44.1 kHz a 2-second clip becomes
$L \approx 11{,}025$ frames.


## SNAKE Activation

**SNAKE** (Liu et al., 2022) adds a learnable sinusoidal term that preserves the
periodic structure present in harmonic audio:

$$\text{SNAKE}(x; \alpha) = x + \frac{1}{\alpha}\sin^2(\alpha x)$$

$\alpha$ is initialized to $\mathbf{1}$ (small perturbation at startup) and
learned per channel. SNAKE is used inside every feed-forward sub-layer in the
transformer blocks.  Because $\alpha$ broadcasts along all leading dimensions,
`Snake` works with any number of batch axes.

```python
class Snake(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        self.alpha = nnx.Param(jnp.ones(features))

    def __call__(self, x: Float[Array, "B F"]) -> Float[Array, "B F"]:
        a = self.alpha.value
        return x + (1.0 / (a + 1e-6)) * jnp.sin(a * x) ** 2
```

```python
def feedforward(dim, ff_dim, rngs: nnx.Rngs):
    return nnx.Sequential(
            nnx.Linear(dim, ff_dim, rngs=rngs),
            Snake(ff_dim, rngs=rngs),
            nnx.Linear(ff_dim, dim, rngs=rngs))
```

## Rotary Positional Embeddings

Rotary positional embeddings (RoPE) encode position by rotating pairs of
features through position-dependent angles. This gives the transformer
translation-equivariant relative position information without adding
explicit position tokens.

Flax provides `nnx.RoPE`, which precomputes cosine and sine frequency
tables once and stores them as module state. To use it with
`nnx.MultiHeadAttention`, pass `nnx.dot_product_attention_with_rope`
(with the `rope` argument partially applied) as the `attention_fn`:

```python
import functools
```

```python
class TransformerBlock(nnx.Module):
    def __init__(
        self, dim: int, num_heads: int, ff_dim: int,
        max_seq_len: int = 2048, *, rngs: nnx.Rngs
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        head_dim = dim // num_heads
        rope = nnx.RoPE(embedding_size=head_dim, max_seq_len=max_seq_len)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            attention_fn=functools.partial(
                nnx.dot_product_attention_with_rope, rope=rope),
            decode=False,
            rngs=rngs,
        )
        self.ff = feedforward(dim, ff_dim, rngs=rngs)
        self.scale1 = nnx.Param(jnp.full(dim, 1e-4))
        self.scale2 = nnx.Param(jnp.full(dim, 1e-4))

    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        normed = self.norm1(x)
        attn_out = self.attn(normed)
        x = x + self.scale1[...] * attn_out
        x = x + self.scale2[...] * self.ff(self.norm2(x))
        return x
```

## Stacking Transformer Layers: The Dual-Path Approach

Full self-attention over $L \approx 11{,}000$ frames costs $O(L^2)$. The
dual-path trick (Luo & Mesgarani, 2020) splits this into two $O(L \cdot K)$
passes:

1. **Intra-chunk** — reshape to $(\ldots, M, K, C)$; each of the $M$ chunks
   attends within itself. Captures local patterns. Cost: $O(M \cdot K^2)$.
2. **Inter-chunk** — swap to $(\ldots, K, M, C)$; each time-slot attends
   across all $M$ chunks. Propagates global pitch/rhythm. Cost: $O(K \cdot M^2)$.

Because `TransformerBlock` accepts `(B, S, D)`, the extra chunk axis
becomes just another batch dimension — no explicit `vmap` is needed.

```python
class DualPathBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        chunk_size: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.intra_block = TransformerBlock(dim, num_heads, ff_dim, rngs=rngs)
        self.inter_block = TransformerBlock(dim, num_heads, ff_dim, rngs=rngs)
        self.chunk_size = chunk_size

    def __call__(self, x: Float[Array, "B L C"]) -> Float[Array, "B L C"]:
        B_shape, L, C = x.shape
        K = self.chunk_size

        # Pad to multiple of chunk_size
        pad_len = (K - L % K) % K
        if pad_len > 0:
            x = jnp.pad(x, [(0, 0)] * len(batch_shape) + [(0, pad_len), (0, 0)])

        L_padded = x.shape[-2]
        M = L_padded // K

        # (B, L, C) -> (B, M, K, C)
        chunks = x.reshape(B_shape, M, K, C)

        # Intra-chunk: TransformerBlock sees (B, M) as batch, K as seq
        chunks = self.intra_block(chunks)

        # Inter-chunk: swap M <-> K, attend, swap back
        inter = jnp.swapaxes(chunks, -3, -2)   # (B, K, M, C)
        inter = self.inter_block(inter)
        chunks = jnp.swapaxes(inter, -3, -2)   # (B, M, K, C)

        out = chunks.reshape(B_shape, L_padded, C)
        return out[..., :L, :]
```

## Splitting into Speaker Streams

After the shared encoder blocks, a `SplitLayer` expands $(\ldots, L, C)$ into
$(\ldots, N, L, C)$.  Splitting here lets each of the $N$ reconstruction stacks
specialize on one speaker while sharing parameters — the subsequent
`DualPathBlock` and `Decoder` layers simply treat the new $N$ axis as an
additional batch dimension.

A GLU gate first refines the shared features before expanding:

$$g, v = \text{split}(W_1 h), \quad h' = \sigma(g) \odot v, \quad \text{streams} = W_2 h' \;\text{reshaped to}\; (\ldots, N, L, C)$$

```python
class SplitLayer(nnx.Module):
    def __init__(self, dim: int, num_stems: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim * 2, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim * num_stems, rngs=rngs)
        self.num_stems = num_stems

    def __call__(self, x: Float[Array, "B L C"]) -> Float[Array, "B N L C"]:
        h = self.linear1(x)                                     # (B, L, 2C)
        gate, val = jnp.split(h, 2, axis=-1)
        h = jax.nn.sigmoid(gate) * val                          # (B, L, C)
        h = self.linear2(h)                                      # (B, L, N*C)
        B_shape, L_dim, _ = h.shape
        C = x.shape[-1]
        h = h.reshape(B_shape, L_dim, self.num_stems, C)   # (B, L, N, C)
        return jnp.swapaxes(h, -3, -2)                          # (B, N, L, C)
```

## Full Forward Pass

After `SplitLayer` produces $(\ldots, N, L, C)$, the reconstruction blocks
and decoder see $(B, N)$ as batch dimensions.  The entire forward pass
runs without any explicit `vmap`.

```python
class SepReformer(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        num_sep_blocks = 2
        num_rec_blocks = 2
        dim = 256
        num_heads = 8
        ff_dim = 1024
        chunk_size = 64

        self.encoder = Encoder(dim, rngs=rngs)
        self.decoder = Decoder(dim, rngs=rngs)
        self.split = SplitLayer(dim, 5, rngs=rngs)
        self.sep_blocks = [
            DualPathBlock(dim, num_heads, ff_dim, chunk_size, rngs=rngs)
            for _ in range(num_sep_blocks)
        ]
        self.rec_blocks = [
            DualPathBlock(dim, num_heads, ff_dim, chunk_size, rngs=rngs)
            for _ in range(num_rec_blocks)
        ]

    def __call__(self, x: Float[Array, "B T"]) -> Float[Array, "B N T"]:
        h = self.encoder(x)                  # (B, L, C)
        for block in self.sep_blocks:
            h = block(h)                     # (B, L, C)
            stems = self.split(h)                # (B, N, L, C)
        for block in self.rec_blocks:
            stems = block(stems)             # (B, N, L, C)  — N is a batch dim
            out = self.decoder(stems)            # (B, N, T')
        # trim / pad to original length
        T = x.shape[-1]
        if out.shape[-1] > T:
            out = out[..., :T]
        elif out.shape[-1] < T:
            pad_width = [(0, 0)] * (out.ndim - 1) + [(0, T - out.shape[-1])]
            out = jnp.pad(out, pad_width)
        return out                           # (B, N, T)

model = SepReformer(rngs=nnx.Rngs(0))
```


## Loss Functions

Supervising a source separator is not straightforward. A plain mean-squared
error (MSE) in the waveform domain penalises tiny timing offsets and global
loudness differences equally, so the model spends capacity chasing irrelevant
phase shifts rather than learning to separate voices. We instead use two
complementary objectives — one waveform-domain and one spectral — that together
give stable, perceptually meaningful gradients.

### SI-SDR

SI-SDR projects the estimate onto the target and reports the energy ratio in dB.
It is invariant to global loudness, which matters for a cappella where voices
differ widely in level:

$$\hat{s}_\text{tgt} = \frac{\langle \hat{s}, s \rangle}{\|s\|^2} s, \qquad \text{SI-SDR} = 10\log_{10}\frac{\|\hat{s}_\text{tgt}\|^2}{\|\hat{s} - \hat{s}_\text{tgt}\|^2}$$

The projection step removes any DC offset before computing the ratio, so a
perfectly separated signal that is merely scaled up or down still scores the
maximum possible value.  In practice, SI-SDR values above $+10\ \text{dB}$
indicate clearly separated sources; below $0\ \text{dB}$ the estimate is
dominated by leakage from other voices.  We negate it to turn maximisation into
minimisation.

```python
def si_sdr(estimate: Float[Array, "T"], target: Float[Array, "T"], eps: float = 1e-8) -> Float[Array, ""]:
    estimate = estimate - jnp.mean(estimate)
    target   = target   - jnp.mean(target)
    dot      = jnp.sum(estimate * target)
    s_target = (dot / (jnp.sum(target ** 2) + eps)) * target
    e_noise  = estimate - s_target
    return 10.0 * jnp.log10(jnp.sum(s_target ** 2) / (jnp.sum(e_noise ** 2) + eps) + eps)
```

### Multi-Resolution STFT Loss

SI-SDR is blind to spectral texture: two signals can have the same SI-SDR yet
sound very different if one has unnatural resonances or missing harmonics.
Adding a frequency-domain term at three FFT scales $\{512, 1024, 2048\}$
addresses this at multiple time-frequency resolutions simultaneously.

A small FFT ($512$) gives sharp time resolution — useful for detecting onset
smearing — while a large FFT ($2048$) gives fine frequency resolution — useful
for resolving individual harmonics in a choir.  Using all three averages out
the inherent time-frequency tradeoff of any single STFT.

Each scale contributes two terms:

- **Spectral convergence** — the Frobenius-norm distance between magnitude
  spectrograms, normalised by the target energy.  This drives the gross shape
  of the spectrum towards the reference.
- **Log-magnitude distance** — the mean absolute difference on a log scale.
  Because human pitch perception is logarithmic, this term penalises errors in
  quiet harmonics just as strongly as errors in loud ones.

$$\mathcal{L}_\text{STFT} = \frac{1}{3}\sum_\text{scale}\left(\underbrace{\frac{\||S| - |\hat{S}|\|_F}{\||S|\|_F}}_{\text{spectral convergence}} + \underbrace{\text{mean}|\log|S| - \log|\hat{S}||}_{\text{log-magnitude}}\right)$$

```python
def stft_mag(x: Float[Array, "T"], fft_size: int, hop: int, win_size: int) -> Float[Array, "F K"]:
    window = jnp.hanning(win_size)
    x_pad  = jnp.pad(x, (fft_size // 2, fft_size // 2))
    n_frames = (len(x_pad) - win_size) // hop + 1
    idx    = jnp.arange(win_size)[None, :] + jnp.arange(n_frames)[:, None] * hop
    frames = x_pad[idx] * window
    return jnp.abs(jnp.fft.rfft(frames, n=fft_size, axis=-1)).T  # (F, K)

def stft_loss_single(est: Float[Array, "T"], tgt: Float[Array, "T"], fft_size: int, hop: int, win: int) -> Float[Array, ""]:
    em, tm = stft_mag(est, fft_size, hop, win), stft_mag(tgt, fft_size, hop, win)
    sc = jnp.linalg.norm(tm - em) / (jnp.linalg.norm(tm) + 1e-8)
    lm = jnp.mean(jnp.abs(jnp.log(em + 1e-8) - jnp.log(tm + 1e-8)))
    return sc + lm

def mr_stft_loss(est: Float[Array, "T"], tgt: Float[Array, "T"]) -> Float[Array, ""]:
    scales = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]
    return sum(stft_loss_single(est, tgt, *s) for s in scales) / len(scales)
```

### Composite Loss

The final objective combines the two terms, with the STFT loss weighted at
$0.5$ so that SI-SDR — which operates directly in the waveform domain and
carries the strongest perceptual signal — dominates early in training.  The
STFT term then fills in spectral detail that SI-SDR cannot see.  Both terms are
averaged across the $N$ stems before being averaged across the batch.

Because the model already handles the batch dimension, `loss_fn` calls the
model once on the full `(B, T)` mixture and then vmaps the per-stem loss
over the batch.

```python
def composite_loss(estimates: Float[Array, "N T"], targets: Float[Array, "N T"]) -> Float[Array, ""]:
    def pair(est, tgt):
        return -si_sdr(est, tgt) + 0.5 * mr_stft_loss(est, tgt)
    return jnp.mean(jax.vmap(pair)(estimates, targets))

def loss_fn(model, mixture: Float[Array, "B T"], targets: Float[Array, "B N T"]) -> Float[Array, ""]:
    estimates = model(mixture)   # (B, N, T)
    return jnp.mean(jax.vmap(composite_loss)(estimates, targets))
```

## Overfitting on JaCappella

Before training on the full corpus, we overfit on a single batch.  This is a
fast sanity check: if the model cannot memorise even one example, something is
wrong with the architecture, the loss, or the data pipeline.  It is much
cheaper to discover this now than after a multi-hour training run.

### Audio Logging

The loss curve tells you the model is learning, but it does not tell you *what*
it is learning.  Listening to the actual estimates at checkpoints is
irreplaceable: you can hear immediately whether the model is separating voices,
producing silence, or emitting noise.  `log_audio_samples` writes one batch of
audio to TensorBoard — the raw mixture, each ground-truth stem, and the
corresponding model estimate — all normalised to a peak of $0.99$ so playback
levels are comparable across steps.

```python
def log_audio_samples(model, loader, writer, global_step):
    mixture, stems = next(iter(loader))
    mix_np = np.array(mixture[0])
    stems_np = np.array(stems[0])
    est_np = np.array(model(mixture[0:1])[0])  # keep batch dim, then index out
    scale  = 0.99 / (np.max(np.abs(mix_np)) + 1e-8)
    writer.add_audio("mixture", mix_np * scale, global_step, sample_rate=SAMPLE_RATE)
    for n in range(stems_np.shape[0]):
        writer.add_audio(f"true/{n}", stems_np[n] * scale, global_step, sample_rate=SAMPLE_RATE)
    for n in range(est_np.shape[0]):
        est_scale  = 0.99 / (np.max(np.abs(est_np[n])) + 1e-8)
        writer.add_audio(f"estimate/{n}", est_np[n] * est_scale, global_step, sample_rate=SAMPLE_RATE)
```

### Optimizer and Training Loop

We use AdamW with a global gradient-norm clip of $1.0$.  Clipping is important
here because early in training the split layer and decoder produce near-random
outputs, which can generate very large gradients through the SI-SDR loss.
Weight decay of $10^{-2}$ provides mild regularisation to prevent any single
stem stream from collapsing to zero.

The loop runs for 50 epochs, logging the scalar loss every 50 steps and
uploading a fresh set of audio samples at the end of each epoch.  You can
monitor progress in TensorBoard with `tensorboard --logdir runs/overfit`.

```python
@nnx.jit
def step(model, optimizer, mixture, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, mixture, targets)
    optimizer.update(model, grads)
    return loss
```

```python
import optax

optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=1e-2)))

writer = SummaryWriter("runs/overfit")
for epoch in range(200):
    for mixture, targets in loader[:1]:
        loss = step(model, optimizer, mixture, targets)
        writer.add_scalar("loss", float(loss), epoch)
        if epoch % 20 == 0:
            log_audio_samples(model, loader, writer, epoch)
writer.close()
```

## Summary

This tutorial walked through building a choral source separator from scratch in Flax NNX. We covered the key ingredients: a convolutional encoder/decoder pair for moving between waveforms and latent frames, SNAKE activations for preserving harmonic structure, dual-path transformer blocks for efficient long-sequence attention, and a GLU-gated split layer for dividing shared representations into per-voice streams. On the training side, we combined SI-SDR and multi-resolution STFT losses to give the model both waveform-level and spectral supervision.

To go beyond overfitting, we'd need to make a few adjustments:
- 2 second segments don't give quite enough context: the original paper uses 4 second audio segments. 
- Data augmentation is essential. In each batch, we can pick a subset of the voice parts to include, and learn to separate just their sum rather than the full mixture. We can also adjust how load each voice part is, or add in convolutional reverb to mimick different acoustics.
- Our model dimensions are simplified compared to the real SepReformer. The paper uses a stride of 4 (vs. our 8), giving twice as many latent frames and finer time resolution. It projects the 256-channel encoder output down to 128 dimensions before the transformer blocks, and runs 4 separator stages (vs. our 2). It also interleaves local convolutional blocks (kernel size 65) with global multi-head attention rather than using pure dual-path transformers, and adds dropout (0.05) throughout. Scaling up to these settings would improve separation quality at the cost of more compute.
