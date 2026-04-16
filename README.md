# ffmpeg-spacemit

FFmpeg fork adding hardware-accelerated encoding, decoding, and
signal-processing for the **SpacemiT K1** SoC (RISC-V RV64GCV),
modeled after [ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip).

## Branches

| Branch | Description |
|--------|-------------|
| `master` | V4L2 M2M hardware codec support (H.264 / HEVC / VP8 / VP9) |
| `k1` | **master + RVV acceleration + V2D hardware scaler** (see below) |

---

## `k1` branch — what's different from `master`

The `k1` branch adds three acceleration layers on top of the V4L2 M2M
codecs from `master`.  All changes compile cleanly with
`--enable-spacemit_mpp`; no extra configure flags are needed.

### 1. RISC-V Vector (RVV) acceleration

Ten hand-written RVV assembly kernels replace critical hot paths that
previously ran as scalar C code on the K1's 8-core RISC-V cluster.

**libswscale / RVV**

| Kernel | File | Description |
|--------|------|-------------|
| Fast bilinear H-scaler | `hyscale_fast_rvv.S` | RVV fast bilinear horizontal scaler |
| Full H-scaler | `hscale_rvv.S` | RVV full horizontal scaler with filter coefficients |
| Vertical output | `output_rvv.S` | RVV vertical output kernels (yuv2plane1, yuv2planeX) |
| NV12/NV21 UV split | `input_rvv.S` (swscale.c) | RVV UV deinterleave for semi-planar formats |

**libavfilter / RVV**

| Kernel | File | Description |
|--------|------|-------------|
| Overlay blend | `vf_overlay_rvv.S` | Per-pixel alpha composite |
| Yadif deinterlace | `vf_yadif_rvv.S` | Temporal/spatial edge-adaptive deinterlace |
| NLMeans denoise | `vf_nlmeans_rvv.S` | Non-local means weight accumulation |

**libavutil / RVV**

| Kernel | File | Description |
|--------|------|-------------|
| Plane copy | `imgutils_rvv.S` | Bulk `av_image_copy` plane loop |

**K1 hardware-bug workarounds** (applied inside these kernels):

- `vnsra.wi` / `vnclipu.wi` narrowing instructions only write even-indexed
  elements when VL > 1 on K1 silicon.  Affected kernels replace the
  narrowing instruction with a same-width shift + stack-strided gather.
- `vnsra.wi` with certain register combinations raises SIGILL; replaced
  with `vsra.vi` + `vmin.vx` + strided `vlse16.v`.
- Register-group alignment: LMUL=2/4 groups must start at even/quad-aligned
  register indices; several initial assignments were realigned.

All workarounds are covered by checkasm and pass with 5+ random seeds.

---

### 2. V2D hardware blitter (`libavcodec/spacemitv2d`)

The K1 integrates a dedicated **V2D 2D-blitter** (`/dev/v2d_dev`) that can
copy and scale NV12 frames entirely in hardware, freeing the CPU cores for
other work during transcode.

**New files**

| File | Purpose |
|------|---------|
| `libavcodec/spacemitv2d.h` | Public API: `SpacemitV2DCtx`, `SpacemitDMAFrameInfo` |
| `libavcodec/spacemitv2d.c` | V2D blit using two Y8 bitblit tasks (Y plane + UV plane) |

**How it works in the encode path**

```
V4L2 decoder CAPTURE buf  ──VIDIOC_EXPBUF──►  DMA fd (Y)
                                               DMA fd (UV)
                                                    │
                         V2D_AddBitblitTask ◄────────┘
                                                    │
V4L2 encoder OUTPUT buf  ◄──VIDIOC_EXPBUF──────────┘
```

When the decoder populates `frame->opaque` with `SpacemitDMAFrameInfo`
(DMA fds for Y and UV planes), `spacemitmppenc` calls `spacemit_v2d_blit()`
instead of `copy_nv12_to_planes()`.  The CPU is not involved in the data
movement.  Falls back to the CPU path transparently when DMA metadata is
absent.

**Library detection**: `configure` checks for `spacemit/v2d_api.h` and
`libv2d.so` automatically when `--enable-spacemit_mpp` is used.

---

### 3. `vf_scale_spacemit` — transparent V2D scale filter

A drop-in hardware scale filter for FFmpeg filter graphs.  Replaces the
software `scale=` filter with V2D hardware scaling; no special frame type
or device context is required.

**New file**: `libavfilter/vf_scale_spacemit.c`

#### Usage

```bash
# Basic downscale
ffmpeg -i input.mp4 -vf scale_spacemit=1280:720 output.mp4

# Expression syntax (same as scale= filter)
ffmpeg -i input.mp4 -vf scale_spacemit=iw/2:ih/2 output.mp4

# Preserve aspect ratio
ffmpeg -i input.mp4 \
    -vf scale_spacemit=1280:720:force_original_aspect_ratio=decrease \
    output.mp4

# Full HW pipeline: HW decode → V2D scale → HW encode
ffmpeg -c:v h264_spacemit_mpp -i input.h264 \
    -vf scale_spacemit=1280:720 \
    -c:v h264_spacemit_mpp -b:v 4M output.mp4
```

#### How it works

```
AVFrame (NV12/YUV420P)
        │
  memcpy to DMA-heap src buffer   ← /dev/dma_heap/system
        │
  V2D_AddBlendTask (NV12 native)  ← hardware bilinear scale
        │
  memcpy from DMA-heap dst buffer
        │
AVFrame (NV12, out_w × out_h)
```

DMA buffers are pre-allocated at `config_output` time and reused every
frame.  Reallocation only occurs if input dimensions change.  Falls back
silently to libswscale if `/dev/v2d_dev` is unavailable.

#### Performance (1080p H.264 decode + scale, K1)

| Filter | Output | fps | Realtime | vs swscale |
|--------|--------|-----|----------|------------|
| `scale=` (swscale) | 1080p | 106 | 3.57× | — |
| `scale_spacemit=` (V2D) | 1080p copy | 85 | 2.84× | −20% (copy overhead) |
| `scale=` (swscale) | 720p | 57 | 1.92× | — |
| `scale_spacemit=` (V2D) | 720p | **85** | **2.86×** | **+49%** |
| `scale=` (swscale) | 480p | 61 | 2.05× | — |
| `scale_spacemit=` (V2D) | 480p | **89** | **3.00×** | **+46%** |
| `scale=` (swscale) | 360p | 67 | 2.25× | — |
| `scale_spacemit=` (V2D) | 360p | **90** | **3.02×** | **+34%** |

> Note: same-size copy is slower via V2D due to DMA staging overhead.
> V2D wins whenever actual scaling is performed.
---

## About (`master` baseline)

The SpacemiT K1 exposes its MVX/Linlon video IP through the standard
Linux **V4L2 Stateful M2M** interface (`V4L2_CAP_VIDEO_M2M_MPLANE`).
No proprietary userspace library is required — the codec talks directly
to the kernel driver via standard POSIX V4L2 ioctls.

## Hardware support

| Codec | Encode | Decode |
|-------|--------|--------|
| H.264 | `h264_spacemit_mpp` | `h264_spacemit_mpp` |
| HEVC  | `hevc_spacemit_mpp` | `hevc_spacemit_mpp` |
| VP8   | `vp8_spacemit_mpp`  | `vp8_spacemit_mpp`  |
| VP9   | `vp9_spacemit_mpp`  | `vp9_spacemit_mpp`  |

## Prerequisites

- SpacemiT K1 board (BPI-F3, Milk-V Jupiter, etc.) running Bianbu OS or
  any Linux with kernel 6.1 and the `mvx` / `linlon` V4L2 driver loaded
- FFmpeg build dependencies: `build-essential nasm pkg-config`
- (`k1` branch only) `/usr/lib/libv2d.so` + `/usr/include/spacemit/v2d_api.h`
  from the `k1x-v2d` system package

## Building

```bash
git clone -b k1 https://github.com/perise/ffmpeg-spacemit.git
cd ffmpeg-spacemit

# --enable-spacemit_mpp auto-enables all 8 codecs + V2D when detected
./configure --enable-spacemit_mpp --prefix=/usr/local

make -j$(nproc)
sudo make install
```

For the K1 native toolchain (clang + RVV march):

```bash
./configure \
    --cc=clang --cxx=clang++ \
    --arch=riscv64 --target-os=linux --cpu=native \
    --extra-cflags='-march=rv64gcv_xsmtvdot -O2' \
    --extra-ldflags=-lspacemit_mpp \
    --enable-spacemit_mpp --enable-v4l2-m2m \
    --enable-gpl --enable-nonfree \
    --disable-vulkan
```

## Usage

### Hardware encode / decode

```bash
# H.264 HW encode
ffmpeg -i input.mp4 -c:v h264_spacemit_mpp -b:v 4M output.mp4

# HEVC HW encode
ffmpeg -i input.mp4 -c:v hevc_spacemit_mpp -b:v 2M output.mp4

# Full HW transcode (decode → scale → encode)
ffmpeg -c:v h264_spacemit_mpp -i input.h264 \
       -vf scale_spacemit=1280:720 \
       -c:v h264_spacemit_mpp -b:v 4M output.mp4
```

### V2D hardware scale filter

```bash
# Drop-in replacement for -vf scale=
ffmpeg -i input.mp4 -vf scale_spacemit=1280:720 output.mp4
ffmpeg -i input.mp4 -vf scale_spacemit=iw/2:ih/2 output.mp4
```

## Reference

- [ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip) — structural reference
- [SpacemiT K1 TRM](https://developer.spacemit.com) — hardware documentation
- [Bianbu OS](https://bianbu.spacemit.com) — official K1 Linux distribution
- [V4L2 stateful codec API](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/dev-decoder.html)
- [RISC-V V spec](https://github.com/riscv/riscv-v-spec) — RVV ISA reference
