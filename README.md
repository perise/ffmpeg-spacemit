# ffmpeg-spacemit

FFmpeg fork adding hardware-accelerated encoding and decoding for the
**SpacemiT K1** SoC (RISC-V RV64GCV), modeled after
[ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip).

## About

The SpacemiT K1 exposes its MVX/Linlon video IP through the standard
Linux **V4L2 Stateful M2M** interface (`V4L2_CAP_VIDEO_M2M_MPLANE`).
No proprietary userspace library is required \xe2\x80\x94 the codec talks directly
to the kernel driver via standard POSIX V4L2 ioctls.

This makes the diff minimal and easy to rebase against future FFmpeg
releases, keeping the project in sync with mainline.

## Hardware support

| Codec | Encode | Decode |
|-------|--------|--------|
| H.264 | `h264_spacemit_mpp` | `h264_spacemit_mpp` |
| HEVC  | `hevc_spacemit_mpp` | `hevc_spacemit_mpp` |
| VP8   | `vp8_spacemit_mpp`  | `vp8_spacemit_mpp`  |
| VP9   | `vp9_spacemit_mpp`  | `vp9_spacemit_mpp`  |

## Prerequisites

- SpacemiT K1 board (BPI-F3, Milk-V Jupiter, etc.) running Bianbu OS or
  any Linux with kernel \xe2\x89\xa5 6.1 and the `mvx` / `linlon` V4L2 driver loaded
- FFmpeg build dependencies: `build-essential nasm pkg-config`

## Building

```bash
# On the K1 board (native)
git clone https://github.com/perise/ffmpeg-spacemit.git
cd ffmpeg-spacemit

# --enable-spacemit_mpp automatically enables all 8 codecs
# (h264/hevc/vp8/vp9 encoder + decoder)
./configure --enable-spacemit_mpp --prefix=/usr/local

make -j$(nproc)
sudo make install
```

To enable only specific codecs instead:

```bash
./configure \
    --enable-encoder=h264_spacemit_mpp \
    --enable-encoder=hevc_spacemit_mpp \
    --enable-decoder=h264_spacemit_mpp \
    --enable-decoder=hevc_spacemit_mpp
```

## Usage

### Encoding

```bash
# H.264 encode (yuv420p or nv12 input accepted)
ffmpeg -i input.mp4 -c:v h264_spacemit_mpp -b:v 4M output.mp4

# HEVC encode
ffmpeg -i input.mp4 -c:v hevc_spacemit_mpp -b:v 2M output.mp4
```

### Decoding

```bash
# H.264 decode (outputs nv12)
ffmpeg -c:v h264_spacemit_mpp -i input.h264 output.yuv

# Transcode: HW decode + HW encode
ffmpeg -c:v h264_spacemit_mpp -i input.h264 \
       -c:v hevc_spacemit_mpp -b:v 2M output.hevc
```

### Options

Both encoder and decoder accept:

| Option | Default | Description |
|--------|---------|-------------|
| `device` | (auto) | V4L2 device path, e.g. `/dev/video0` |
| `num_output_bufs` | 4/8 | V4L2 OUTPUT queue buffer count |
| `num_capture_bufs` | 4/8 | V4L2 CAPTURE queue buffer count |

## Keeping up to date with FFmpeg mainline

All SpacemiT-specific changes are either new files or additive lines in
`configure`, `libavcodec/Makefile`, and `libavcodec/allcodecs.c`, so
rebasing is straightforward:

```bash
git remote add upstream https://github.com/ffmpeg/ffmpeg.git
git fetch upstream
git rebase upstream/master
git push --force-with-lease origin master
```

## Why not use the existing `h264_v4l2m2m` encoder?

FFmpeg\xe2\x80\x99s built-in V4L2 M2M code defaults to single-planar buffer types.
The SpacemiT K1 MVX driver requires `_MPLANE` types for **all** ioctls
and only reports `V4L2_CAP_VIDEO_M2M_MPLANE`. `spacemitmppenc` /
`spacemitmppdec` always use `V4L2_BUF_TYPE_VIDEO_*_MPLANE` and call
`VIDIOC_G_FMT` after `VIDIOC_S_FMT` to read back the actual `num_planes`
assigned by the driver. The encoder also handles inline YUV420P\xe2\x86\x92NV12
conversion so standard pixel formats work without an extra `scale` step.

## Reference

- [ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip)   structural reference
- [SpacemiT K1 TRM](https://developer.spacemit.com)   hardware documentation
- [Bianbu OS](https://bianbu.spacemit.com)   official K1 Linux distribution
- [V4L2 stateful codec API](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/dev-decoder.html)
