# audiolocate

> *This document was AI-translated from the [Traditional Chinese original](https://github.com/JoeYang1412/audiolocate/blob/master/README_zhTW.md).*

An audio fingerprint matching engine based on the algorithm proposed in Wang (2003) "An Industrial-Strength Audio Search Algorithm." Given a short audio sample and a long reference audio, it determines whether the sample appears in the reference audio and returns the precise time offset.

## Features

- **Spectral Peak Constellation Fingerprinting** — Generates hashes via STFT spectral local peak pairing for efficient matching
- **Noise Resilience** — Correctly identifies matches even under 3 dB SNR conditions
- **Fast Localization in Long Audio** — Chunked streaming with early-exit mechanism stops on hit, no need to scan the entire reference audio, significantly reducing matching time for long audio
- **Streaming Processing** — Supports chunked streaming matching for large audio files with controllable memory usage
- **Parallel Decoding** — Audio decoding and fingerprint computation run concurrently for improved throughput
- **Multi-Format Support** — Supports common audio formats such as MP3, AAC, and WAV via PyAV
- **Multiple Input Sources** — Accepts file paths, URLs, and file-like objects

## Installation

```bash
pip install audiolocate
```

Or install from source:

```bash
git clone https://github.com/JoeYang1412/audiolocate.git
cd audiolocate
pip install .
```

## Quick Start

### Basic Matching

```python
from audiolocate import AudioFingerprint

fp = AudioFingerprint()
result = fp.find_match("reference.wav", "sample.wav")

if result["found"]:
    print(f"Match found at {result['time_seconds']:.2f} seconds")
else:
    print("No match found")
```

### Streaming Matching (for Large Files)

```python
from audiolocate import StreamMatcher

matcher = StreamMatcher()
result = matcher.find_match_from_sources(
    "long_audio.wav",
    "short_sample.wav",
    chunk_seconds=300,   # Process 300 seconds per chunk
    early_exit=True,     # Stop immediately after finding a match
    verbose=True         # Show processing progress
)

if result["found"]:
    print(f"Match found at {result['time_seconds']:.2f} seconds")
    print(f"Processed {result['chunks_processed']} chunks")
```

In practice, locating a 10-second sample within a 4-hour reference audio: when the match point is near the beginning, early-exit hits on the first batch of chunks in approximately **10 seconds**; when the match point is at the very end and all chunks must be scanned, it completes in approximately **45 seconds** (including network I/O, decoding, and fingerprint computation).

### Custom Parameters

```python
fp = AudioFingerprint(
    sr=16000,                # Sample rate (default: 8000)
    n_fft=2048,              # FFT window size (default: 1024)
    peaks_per_second=50,     # Number of peaks per second (default: 30)
    significance_factor=2.5  # Statistical significance threshold factor (default: 3.0)
)
```

## Classes and Methods

### `AudioFingerprint`

Core fingerprint matching class.

| Method | Description |
|--------|-------------|
| `load_audio(source)` | Load audio and resample to mono |
| `fingerprint_audio(audio)` | Generate audio fingerprint (hash dictionary + frame count) |
| `find_match(reference, sample)` | High-level API: match two audio sources and return a result dictionary |
| `detect_peaks(spectrogram)` | Detect local peaks in a spectrogram |
| `generate_hashes(peaks)` | Generate paired hashes from peak constellation |
| `match(db_hashes, sample_hashes)` | Match two hash sets via offset histogram |

### `StreamMatcher`

Inherits from `AudioFingerprint`, providing streaming processing capabilities.

| Method | Description |
|--------|-------------|
| `find_match_from_sources(long_source, short_source, ...)` | Main entry point for streaming matching |
| `fingerprint_source(source, chunk_seconds=300)` | Build fingerprint via chunked streaming |

### `MatchResult`

Match result (NamedTuple).

| Field | Type | Description |
|-------|------|-------------|
| `is_match` | `bool` | Whether a match was found |
| `offset_frames` | `int` | Frame offset of the match position |
| `match_count` | `int` | Number of matching hashes |
| `threshold` | `float` | Adaptive threshold used |
| `noise_baseline` | `float` | Computed noise baseline |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sr` | 8000 | Sample rate (Hz) |
| `n_fft` | 1024 | FFT window size |
| `hop_length` | 256 | STFT hop length |
| `peak_neighborhood_size` | 15 | Peak detection neighborhood size |
| `peaks_per_second` | 30 | Target number of peaks per second |
| `fan_value` | 10 | Pairing fan-out per anchor point |
| `target_t_min` / `target_t_max` | 2 / 100 | Pairing time difference range (frames) |
| `target_f_min` / `target_f_max` | -30 / 60 | Pairing frequency difference range (bins) |
| `significance_factor` | 3.0 | Statistical threshold multiplier |
| `prominence_factor` | 2.0 | Peak prominence multiplier |
| `min_count_floor` | 5 | Minimum match count threshold |


## Algorithm Overview

1. **STFT** — Convert audio into a time-frequency spectrogram
2. **Peak Detection** — Extract constellation using local maximum filter with block density control
3. **Hash Generation** — Encode peak pairs as 32-bit hashes (frequency pair + time difference)
4. **Offset Histogram** — Count hash hits for each time offset between sample and reference
5. **Adaptive Thresholding** — Three-layer determination combining minimum threshold, statistical significance, and prominence

## References

> Wang, A. (2003). An Industrial-Strength Audio Search Algorithm.
> In *Proceedings of the 4th International Conference on Music
> Information Retrieval (ISMIR)*.

## License

See the license file in the project root directory.
