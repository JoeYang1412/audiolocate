import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import stft as scipy_stft
from collections import defaultdict, Counter
from typing import NamedTuple, Dict, List, Tuple, Union
import pathlib


class MatchResult(NamedTuple):
    is_match: bool
    offset_frames: int
    match_count: int
    threshold: float
    noise_baseline: float


class AudioFingerprint:
    """Audio fingerprinting engine based on Wang 2003 (Shazam algorithm)."""

    def __init__(
        self,
        sr: int = 8000,
        n_fft: int = 1024,
        hop_length: int = 256,
        peak_neighborhood_size: int = 15,
        peaks_per_second: float = 30,
        time_block_seconds: float = 1.0,
        fan_value: int = 10,
        target_t_min: int = 2,
        target_t_max: int = 100,
        target_f_min: int = -30,
        target_f_max: int = 60,
        significance_factor: float = 3.0,
        prominence_factor: float = 2.0,
        min_count_floor: int = 5,
        freq_bits: int = 10,
        dt_bits: int = 12,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.peak_neighborhood_size = peak_neighborhood_size
        self.peaks_per_second = peaks_per_second
        self.time_block_seconds = time_block_seconds
        self.fan_value = fan_value
        self.target_t_min = target_t_min
        self.target_t_max = target_t_max
        self.target_f_min = target_f_min
        self.target_f_max = target_f_max
        self.significance_factor = significance_factor
        self.prominence_factor = prominence_factor
        self.min_count_floor = min_count_floor
        self.freq_bits = freq_bits
        self.dt_bits = dt_bits

        # Derived constants
        self.frames_per_second = sr / hop_length
        self.time_block_frames = int(time_block_seconds * self.frames_per_second)
        self.peaks_per_block = int(peaks_per_second * time_block_seconds)
        self.freq_mask = (1 << freq_bits) - 1
        self.dt_mask = (1 << dt_bits) - 1

    def load_audio(self, path: str) -> np.ndarray:
        """Load an audio file using PyAV, convert to mono float32 at self.sr."""
        import av

        container = av.open(str(path))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.sr)
        chunks = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                chunks.append(r.to_ndarray().flatten())
        # Flush resampler (spec §8.7)
        flushed = resampler.resample(None)
        for r in flushed:
            chunks.append(r.to_ndarray().flatten())
        container.close()
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def _compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrogram using scipy.signal.stft.

        Returns linear magnitude, shape (n_fft//2+1, n_frames), float32.
        Uses boundary=None, padded=False for center=False behavior (spec §8.5).
        """
        if len(audio) < self.n_fft:
            return np.zeros((self.n_fft // 2 + 1, 0), dtype=np.float32)
        _, _, Zxx = scipy_stft(
            audio,
            fs=self.sr,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=None,
            padded=False,
        )
        return np.abs(Zxx).astype(np.float32)

    def detect_peaks(self, spectrogram: np.ndarray) -> List[Tuple[int, int]]:
        """Detect peaks in spectrogram using local maximum filter + density control.

        Returns list of (freq_bin, time_frame) sorted by time.
        """
        if spectrogram.shape[1] == 0:
            return []

        # Local maximum detection (spec §2.1 step 2)
        local_max = maximum_filter(
            spectrogram,
            size=(self.peak_neighborhood_size, self.peak_neighborhood_size),
        )
        is_peak = (spectrogram == local_max) & (spectrogram > 0)

        # Get all candidate peak coordinates and amplitudes
        freq_bins, time_frames = np.where(is_peak)
        if len(freq_bins) == 0:
            return []
        amplitudes = spectrogram[freq_bins, time_frames]

        # Density control: block-wise top-K (spec §2.1 step 3)
        # Assign each candidate to its block via integer division, then sort
        # to group by block. This avoids scanning all candidates per block
        # (O(num_blocks × num_candidates) → O(num_candidates log num_candidates)).
        block_ids = time_frames // self.time_block_frames
        order = np.argsort(block_ids, kind='stable')
        sorted_block_ids = block_ids[order]
        unique_blocks = np.unique(sorted_block_ids)
        b_starts = np.searchsorted(sorted_block_ids, unique_blocks)
        b_ends = np.searchsorted(sorted_block_ids, unique_blocks, side='right')

        selected = []
        for b in range(len(unique_blocks)):
            block_slice = order[b_starts[b]:b_ends[b]]
            k = min(self.peaks_per_block, len(block_slice))
            block_amps = amplitudes[block_slice]
            if k < len(block_slice):
                top_k_local = np.argpartition(block_amps, -k)[-k:]
            else:
                top_k_local = np.arange(len(block_slice))
            selected.append(block_slice[top_k_local])

        if not selected:
            return []
        selected = np.concatenate(selected)

        # Build result sorted by time then frequency
        peaks = list(zip(freq_bins[selected].tolist(), time_frames[selected].tolist()))
        peaks.sort(key=lambda p: (p[1], p[0]))
        return peaks

    def _pack_hash(self, f1: int, f2: int, dt: int) -> int:
        """Bit-pack (f1, f2, dt) into a 32-bit hash."""
        return (
            ((f1 & self.freq_mask) << (self.freq_bits + self.dt_bits))
            | ((f2 & self.freq_mask) << self.dt_bits)
            | (dt & self.dt_mask)
        )

    def generate_hashes(self, peaks: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """Generate combinatorial hashes from constellation peaks.

        For each anchor, searches forward independently from i+1 (spec §8.3).
        Uses searchsorted + vectorized filtering to avoid pure-Python inner loop.
        Returns dict mapping 32-bit hash -> list of anchor time offsets.
        """
        if not peaks:
            return {}

        pts = np.array(peaks, dtype=np.int64)
        freqs = pts[:, 0]
        times = pts[:, 1]
        n = len(pts)

        # Precompute candidate range bounds via binary search
        j_starts = np.searchsorted(times, times + self.target_t_min)
        j_ends = np.searchsorted(times, times + self.target_t_max, side='right')

        shift = self.freq_bits + self.dt_bits
        freq_mask = self.freq_mask
        dt_bits = self.dt_bits
        dt_mask = self.dt_mask
        fan = self.fan_value
        f_min = self.target_f_min
        f_max = self.target_f_max

        all_keys = []
        all_vals = []
        for i in range(n):
            # Ensure we never include anchor itself or earlier peaks (spec §8.3)
            js = max(j_starts[i], i + 1)
            je = j_ends[i]
            if js >= je:
                continue
            cand_f = freqs[js:je]
            df = cand_f - freqs[i]
            mask = (df >= f_min) & (df <= f_max)
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            if len(idx) > fan:
                idx = idx[:fan]
            f2 = cand_f[idx]
            dt = times[js:je][idx] - times[i]
            h = ((freqs[i] & freq_mask) << shift
                 | (f2 & freq_mask) << dt_bits
                 | (dt & dt_mask))
            all_keys.append(h)
            all_vals.append(np.full(len(idx), times[i], dtype=np.int32))

        if not all_keys:
            return {}

        keys = np.concatenate(all_keys)
        vals = np.concatenate(all_vals)

        hash_dict = defaultdict(list)
        for k, v in zip(keys.tolist(), vals.tolist()):
            hash_dict[k].append(v)
        return dict(hash_dict)

    def fingerprint_audio(
        self, audio: np.ndarray
    ) -> Tuple[Dict[int, List[int]], int]:
        """Compute STFT -> detect peaks -> generate hashes.

        Returns (hashes_dict, n_frames).
        """
        spectrogram = self._compute_stft(audio)
        peaks = self.detect_peaks(spectrogram)
        hashes = self.generate_hashes(peaks)
        return hashes, spectrogram.shape[1]

    def _evaluate_offsets(self, offset_counts: dict) -> MatchResult:
        """Evaluate offset histogram with adaptive threshold (spec §2.3)."""
        if not offset_counts:
            return MatchResult(False, 0, 0, float(self.min_count_floor), 0.0)

        best_offset = max(offset_counts, key=offset_counts.get)
        best_count = offset_counts[best_offset]

        # Noise bins: exclude best offset ± tolerance
        tolerance = 2
        noise_counts = [
            c
            for off, c in offset_counts.items()
            if abs(off - best_offset) > tolerance
        ]

        if len(noise_counts) >= 10:
            noise_mean = float(np.mean(noise_counts))
            noise_std = float(np.std(noise_counts))
            noise_max = float(max(noise_counts))
            stat_threshold = noise_mean + self.significance_factor * noise_std
            # Prominence: best must clearly stand out from the highest noise bin.
            # Necessary because offset-count noise is Poisson-like (skewed),
            # and mean+3σ underestimates the tail.
            prominence_threshold = noise_max * self.prominence_factor
            threshold = max(self.min_count_floor, stat_threshold, prominence_threshold)
        else:
            noise_mean = 0.0
            threshold = float(self.min_count_floor)

        is_match = best_count > threshold
        return MatchResult(is_match, best_offset, best_count, threshold, noise_mean)

    def match(
        self,
        db_hashes: Dict[int, List[int]],
        sample_hashes: Dict[int, List[int]],
    ) -> MatchResult:
        """Match sample hashes against reference (db) hashes.

        Builds offset histogram and evaluates with adaptive threshold.
        Uses numpy broadcasting to vectorize pairwise offset computation.
        """
        all_deltas = []
        for h, s_offs in sample_hashes.items():
            d_offs = db_hashes.get(h)
            if d_offs is None:
                continue
            d = np.array(d_offs[:500], dtype=np.int64)
            s = np.array(s_offs[:500], dtype=np.int64)
            all_deltas.append((d[:, None] - s[None, :]).ravel())

        if not all_deltas:
            return self._evaluate_offsets({})

        offset_counts = Counter(np.concatenate(all_deltas).tolist())
        return self._evaluate_offsets(dict(offset_counts))

    def find_match(
        self,
        reference: Union[np.ndarray, str, pathlib.Path],
        sample: Union[np.ndarray, str, pathlib.Path],
    ) -> dict:
        """High-level API: find whether sample appears in reference.

        Accepts numpy arrays or file paths for both arguments.
        Returns dict with keys: found, time_seconds, match_count, threshold, noise_baseline.
        """
        if isinstance(reference, (str, pathlib.Path)):
            reference = self.load_audio(reference)
        if isinstance(sample, (str, pathlib.Path)):
            sample = self.load_audio(sample)

        ref_hashes, _ = self.fingerprint_audio(reference)
        sample_hashes, _ = self.fingerprint_audio(sample)
        result = self.match(ref_hashes, sample_hashes)

        time_seconds = (
            result.offset_frames * self.hop_length / self.sr
            if result.is_match
            else None
        )
        return {
            "found": result.is_match,
            "time_seconds": time_seconds,
            "match_count": result.match_count,
            "threshold": result.threshold,
            "noise_baseline": result.noise_baseline,
        }
