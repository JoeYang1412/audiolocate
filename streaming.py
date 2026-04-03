import math
import threading
import queue as _queue
import time as _time
import numpy as np
from collections import defaultdict, Counter

from .core import AudioFingerprint, MatchResult


def _format_time(seconds):
    """Format seconds into H:MM:SS or M:SS."""
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class StreamMatcher(AudioFingerprint):
    """Streaming audio fingerprint matcher for large files.

    Inherits from AudioFingerprint and adds chunked processing
    so that only one chunk of raw audio is in memory at a time.
    """

    def fingerprint_source(
        self, source: str, chunk_seconds: int = 300, verbose: bool = True
    ) -> dict:
        """Stream-decode an audio source and build fingerprint hash dictionary.

        source can be a file path or any URL accepted by av.open().
        Processes audio in chunks of chunk_seconds to bound memory usage.
        Returns dict mapping 32-bit hash -> list of global frame offsets.
        """
        import av

        container = av.open(str(source))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.sr)

        chunk_samples = int(chunk_seconds * self.sr)
        buf_parts = []
        buf_len = 0
        global_sample_offset = 0
        all_hashes = defaultdict(list)
        chunks_done = 0
        total_hashes = 0

        if verbose:
            print(f"[fingerprint] 開始建立指紋: {source}")

        def _process_chunk(chunk_audio, sample_offset):
            nonlocal chunks_done, total_hashes
            if len(chunk_audio) < self.n_fft:
                return
            spectrogram = self._compute_stft(chunk_audio)
            peaks = self.detect_peaks(spectrogram)
            hashes = self.generate_hashes(peaks)
            frame_offset = sample_offset // self.hop_length
            chunk_hash_count = sum(len(v) for v in hashes.values())
            for h, offsets in hashes.items():
                all_hashes[h].extend(off + frame_offset for off in offsets)
            chunks_done += 1
            total_hashes += chunk_hash_count
            if verbose:
                pos = _format_time(sample_offset / self.sr)
                dur = _format_time(len(chunk_audio) / self.sr)
                print(f"  chunk {chunks_done} | 位置 {pos} | "
                      f"長度 {dur} | peaks {len(peaks)} | "
                      f"hashes +{chunk_hash_count} (累計 {total_hashes})")

        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                buf_parts.append(r.to_ndarray().flatten())
                buf_len += len(buf_parts[-1])
            while buf_len >= chunk_samples:
                buf = np.concatenate(buf_parts)
                chunk = buf[:chunk_samples]
                remainder = buf[chunk_samples:]
                buf_parts = [remainder] if len(remainder) > 0 else []
                buf_len = len(remainder)
                _process_chunk(chunk, global_sample_offset)
                global_sample_offset += chunk_samples

        # Flush resampler (spec §8.7)
        flushed = resampler.resample(None)
        for r in flushed:
            buf_parts.append(r.to_ndarray().flatten())
            buf_len += len(buf_parts[-1])

        # Process remaining buffer
        if buf_len > 0:
            buffer = np.concatenate(buf_parts)
            _process_chunk(buffer, global_sample_offset)

        container.close()

        if verbose:
            print(f"[fingerprint] 完成 | {chunks_done} chunks | "
                  f"{len(all_hashes)} unique hashes | "
                  f"{total_hashes} total entries")

        return dict(all_hashes)

    def find_match_from_sources(
        self,
        long_source: str,
        short_source: str,
        chunk_seconds: int = 300,
        early_exit: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Match a short sample against a long reference using streaming.

        Both long_source and short_source can be file paths or any URL
        accepted by av.open(). The short audio is fingerprinted entirely in
        memory. The long audio is streamed chunk by chunk with a decode
        thread pipelined against the processing thread. Each chunk builds
        an independent offset histogram and is evaluated independently
        (spec §2.3, §8.1). Adjacent chunks overlap by the sample duration
        to avoid boundary-splitting false negatives (spec §8.2).

        Returns dict with keys: found, time_seconds, match_count, threshold,
        noise_baseline, chunks_processed, early_stopped.
        """
        import av

        t_start = _time.monotonic()

        # Step 1: fingerprint the short (sample) audio in memory
        if verbose:
            print(f"[match] 載入樣本: {short_source}")
        short_audio = self.load_audio(short_source)
        sample_duration = len(short_audio) / self.sr
        sample_hashes, _ = self.fingerprint_audio(short_audio)
        total_sample_hashes = sum(len(v) for v in sample_hashes.values())
        del short_audio  # free memory
        if verbose:
            print(f"[match] 樣本長度 {_format_time(sample_duration)} | "
                  f"{len(sample_hashes)} unique hashes | "
                  f"{total_sample_hashes} total entries")

        # Step 2: compute overlap (spec §4.2, §8.2)
        overlap_seconds = math.ceil(sample_duration)
        chunk_samples = int(chunk_seconds * self.sr)
        overlap_samples = int(overlap_seconds * self.sr)
        step_samples = chunk_samples - overlap_samples
        if step_samples <= 0:
            raise ValueError(
                f"chunk_seconds ({chunk_seconds}) must be larger than "
                f"sample duration ({sample_duration:.1f}s)"
            )

        if verbose:
            print(f"[match] 開始串流比對: {long_source} "
                  f"(chunk={chunk_seconds}s, overlap={overlap_seconds}s, "
                  f"early_exit={early_exit})")

        # Step 3: pipelined decode (thread) + process (main thread)
        # PyAV decode and numpy both release GIL, enabling true parallelism.
        chunk_queue = _queue.Queue(maxsize=2)
        stop_event = threading.Event()
        decode_error = [None]  # mutable container for thread exception

        def _put(item):
            """Put item into queue, retrying with timeout until success or stop."""
            while not stop_event.is_set():
                try:
                    chunk_queue.put(item, timeout=0.5)
                    return True
                except _queue.Full:
                    continue
            return False

        def _decode_thread():
            """Producer: decode audio and emit overlapping chunks."""
            try:
                container = av.open(str(long_source))
                resampler = av.AudioResampler(
                    format="fltp", layout="mono", rate=self.sr
                )
                buf_parts = []
                buf_len = 0
                offset = 0

                for frame in container.decode(audio=0):
                    if stop_event.is_set():
                        break
                    resampled = resampler.resample(frame)
                    for r in resampled:
                        part = r.to_ndarray().flatten()
                        buf_parts.append(part)
                        buf_len += len(part)
                    while buf_len >= chunk_samples and not stop_event.is_set():
                        buf = np.concatenate(buf_parts)
                        chunk = buf[:chunk_samples]
                        remainder = buf[step_samples:]
                        buf_parts = (
                            [remainder] if len(remainder) > 0 else []
                        )
                        buf_len = len(remainder)
                        if not _put((chunk, offset)):
                            break
                        offset += step_samples

                if not stop_event.is_set():
                    # Flush resampler (spec §8.7)
                    flushed = resampler.resample(None)
                    for r in flushed:
                        part = r.to_ndarray().flatten()
                        buf_parts.append(part)
                        buf_len += len(part)
                    if buf_len > 0:
                        buf = np.concatenate(buf_parts)
                        _put((buf, offset))

                container.close()
            except Exception as e:
                decode_error[0] = e
            finally:
                # Sentinel must always be sent; use timeout to avoid deadlock
                # if consumer is gone and queue is full.
                for _ in range(10):
                    try:
                        chunk_queue.put(None, timeout=0.5)
                        break
                    except _queue.Full:
                        continue

        chunks_processed = 0
        best_result = None

        def _process_chunk(chunk_audio, sample_offset):
            """Consumer: fingerprint one chunk and match independently."""
            nonlocal chunks_processed, best_result

            if len(chunk_audio) < self.n_fft:
                return None

            t_chunk_start = _time.monotonic()
            spectrogram = self._compute_stft(chunk_audio)
            peaks = self.detect_peaks(spectrogram)
            chunk_hashes = self.generate_hashes(peaks)
            frame_offset = sample_offset // self.hop_length
            chunks_processed += 1

            # Build independent offset histogram (spec §8.1)
            all_deltas = []
            for h, chunk_offsets in chunk_hashes.items():
                s_offs = sample_hashes.get(h)
                if s_offs is None:
                    continue
                c = np.array(chunk_offsets[:500], dtype=np.int64) + frame_offset
                s = np.array(s_offs[:500], dtype=np.int64)
                all_deltas.append((c[:, None] - s[None, :]).ravel())

            if all_deltas:
                concat = np.concatenate(all_deltas)
                hits = len(concat)
                chunk_offset_counts = Counter(concat.tolist())
            else:
                hits = 0
                chunk_offset_counts = {}

            result = self._evaluate_offsets(dict(chunk_offset_counts))

            if best_result is None or result.match_count > best_result.match_count:
                best_result = result

            t_chunk_elapsed = _time.monotonic() - t_chunk_start
            if verbose:
                pos = _format_time(sample_offset / self.sr)
                dur = _format_time(len(chunk_audio) / self.sr)
                ratio = (result.match_count / result.threshold
                         if result.threshold > 0 else 0)
                status = "MATCH" if result.is_match else "---"
                print(f"  chunk {chunks_processed} | 位置 {pos} | "
                      f"長度 {dur} | peaks {len(peaks)} | "
                      f"hits {hits} | "
                      f"best {result.match_count}/{result.threshold:.0f} "
                      f"({ratio:.1f}x) {status} | "
                      f"{t_chunk_elapsed:.1f}s")

            return result

        # Start pipeline
        decoder = threading.Thread(target=_decode_thread, daemon=True)
        decoder.start()

        early_stopped = False
        match_result = None

        while True:
            try:
                item = chunk_queue.get(timeout=5)
            except _queue.Empty:
                # Decode thread may have died without sending sentinel
                if not decoder.is_alive():
                    break
                continue
            if item is None:
                break
            chunk_audio, sample_offset = item
            result = _process_chunk(chunk_audio, sample_offset)

            if early_exit and result is not None and result.is_match:
                match_result = result
                early_stopped = True
                stop_event.set()
                # Drain queue so decode thread can unblock from put()
                while True:
                    try:
                        chunk_queue.get_nowait()
                    except _queue.Empty:
                        break
                break

        decoder.join(timeout=10)

        # Re-raise decode thread exception if any
        if decode_error[0] is not None:
            raise decode_error[0]

        # Return matching result, or best result for diagnostics
        final_result = match_result or best_result
        if final_result is None:
            final_result = MatchResult(
                False, 0, 0, float(self.min_count_floor), 0.0
            )

        final = self._build_result(final_result, chunks_processed, early_stopped)

        if verbose:
            elapsed = _time.monotonic() - t_start
            if final["found"]:
                print(f"[match] 找到匹配！位於 "
                      f"{_format_time(final['time_seconds'])} | "
                      f"count={final['match_count']} "
                      f"threshold={final['threshold']:.1f} | "
                      f"{chunks_processed} chunks | {elapsed:.1f}s")
            else:
                print(f"[match] 未找到匹配 | "
                      f"best count={final['match_count']} "
                      f"threshold={final['threshold']:.1f} | "
                      f"{chunks_processed} chunks | {elapsed:.1f}s")

        return final

    def _build_result(
        self, result, chunks_processed: int, early_stopped: bool
    ) -> dict:
        """Format the match result into an output dictionary."""
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
            "chunks_processed": chunks_processed,
            "early_stopped": early_stopped,
        }
