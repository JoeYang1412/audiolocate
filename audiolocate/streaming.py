import math
import threading
import queue as _queue
import time as _time
import numpy as np
from collections import defaultdict, Counter
from typing import IO, Union
import pathlib

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

    # ── Class Constants ──
    CHUNK_QUEUE_SIZE = 2
    QUEUE_PUT_TIMEOUT = 0.5
    QUEUE_GET_TIMEOUT = 5
    DECODER_JOIN_TIMEOUT = 10
    SENTINEL_RETRY_COUNT = 10

    # ── Fingerprinting (streaming) ──

    def fingerprint_source(
        self, source: Union[str, pathlib.Path, IO[bytes]], chunk_seconds: int = 300, verbose: bool = False
    ) -> dict:
        """Stream-decode an audio source and build fingerprint hash dictionary.

        source can be a file path, URL, or file-like object accepted by av.open().
        Processes audio in chunks of chunk_seconds to bound memory usage.
        Returns dict mapping 32-bit hash -> list of global frame offsets.
        """
        import av

        container = av.open(source)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.sr)

        chunk_samples = int(chunk_seconds * self.sr)
        buf_parts = []
        buf_len = 0
        global_sample_offset = 0
        all_hashes = defaultdict(list)
        chunks_done = 0
        total_hashes = 0

        print(f"[fingerprint] Building fingerprint: {source}")

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
                print(f"  chunk {chunks_done} | pos {pos} | "
                      f"len {dur} | peaks {len(peaks)} | "
                      f"hashes +{chunk_hash_count} (total {total_hashes})")
            else:
                print(f"\r[fingerprint] Processing chunk {chunks_done} ...", end="", flush=True)

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

        if not verbose:
            print()
        print(f"[fingerprint] Done | {chunks_done} chunks | "
              f"{len(all_hashes)} unique hashes | "
              f"{total_hashes} total entries")

        return dict(all_hashes)

    # ── Chunk Matching ──

    def _match_chunk(self, chunk_audio, sample_offset, sample_hashes, chunk_index, verbose):
        """Fingerprint one chunk and match against sample hashes independently.

        Returns (MatchResult, hits) or (None, 0) if chunk is too short.
        """
        if len(chunk_audio) < self.n_fft:
            return None, 0

        t_chunk_start = _time.monotonic()
        spectrogram = self._compute_stft(chunk_audio)
        peaks = self.detect_peaks(spectrogram)
        chunk_hashes = self.generate_hashes(peaks)
        frame_offset = sample_offset // self.hop_length

        # Build independent offset histogram (spec §8.1)
        all_deltas = []
        for hash_key, chunk_offsets in chunk_hashes.items():
            s_offs = sample_hashes.get(hash_key)
            if s_offs is None:
                continue
            chunk_arr = np.array(chunk_offsets[:self.MAX_OFFSETS_PER_HASH], dtype=np.int64) + frame_offset
            sample_arr = np.array(s_offs[:self.MAX_OFFSETS_PER_HASH], dtype=np.int64)
            # Pairwise differences: every chunk offset minus every sample offset
            all_deltas.append((chunk_arr[:, None] - sample_arr[None, :]).ravel())

        if all_deltas:
            concat = np.concatenate(all_deltas)
            hits = len(concat)
            chunk_offset_counts = Counter(concat.tolist())
        else:
            hits = 0
            chunk_offset_counts = {}

        result = self._evaluate_offsets(dict(chunk_offset_counts))

        if verbose:
            t_elapsed = _time.monotonic() - t_chunk_start
            pos = _format_time(sample_offset / self.sr)
            dur = _format_time(len(chunk_audio) / self.sr)
            ratio = result.match_count / result.threshold if result.threshold > 0 else 0
            status = "MATCH" if result.is_match else "---"
            print(f"  chunk {chunk_index} | pos {pos} | "
                  f"len {dur} | peaks {len(peaks)} | "
                  f"hits {hits} | "
                  f"best {result.match_count}/{result.threshold:.0f} "
                  f"({ratio:.1f}x) {status} | "
                  f"{t_elapsed:.1f}s")

        return result, hits

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

    # ── Streaming Pipeline ──

    def find_match_from_sources(
        self,
        long_source: Union[str, pathlib.Path, IO[bytes]],
        short_source: Union[str, pathlib.Path, IO[bytes]],
        chunk_seconds: int = 300,
        early_exit: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Match a short sample against a long reference using streaming.

        Both long_source and short_source can be file paths, URLs, or
        file-like objects accepted by av.open(). The short audio is
        fingerprinted entirely in memory. The long audio is streamed
        chunk by chunk with a decode thread pipelined against the
        processing thread. Each chunk builds
        an independent offset histogram and is evaluated independently
        (spec §2.3, §8.1). Adjacent chunks overlap by the sample duration
        to avoid boundary-splitting false negatives (spec §8.2).

        Returns dict with keys: found, time_seconds, match_count, threshold,
        noise_baseline, chunks_processed, early_stopped.
        """
        import av

        t_start = _time.monotonic()

        # Step 1: fingerprint the short (sample) audio in memory
        print(f"[load] Loading sample: {short_source}")
        short_audio = self.load_audio(short_source)
        sample_duration = len(short_audio) / self.sr
        sample_hashes, _ = self.fingerprint_audio(short_audio)
        total_sample_hashes = sum(len(v) for v in sample_hashes.values())
        del short_audio  # free memory
        if verbose:
            print(f"[load] {_format_time(sample_duration)} | "
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
            source_name = getattr(long_source, 'name', str(long_source))
            print(f"[scan] {source_name} "
                  f"(chunk={chunk_seconds}s, overlap={overlap_seconds}s, "
                  f"early_exit={early_exit})")

        # Step 3: pipelined decode (thread) + process (main thread)
        # PyAV decode and numpy both release GIL, enabling true parallelism.
        chunk_queue = _queue.Queue(maxsize=self.CHUNK_QUEUE_SIZE)
        stop_event = threading.Event()
        decode_error = [None]  # mutable container for thread exception

        def _put(item):
            """Put item into queue, retrying with timeout until success or stop."""
            while not stop_event.is_set():
                try:
                    chunk_queue.put(item, timeout=self.QUEUE_PUT_TIMEOUT)
                    return True
                except _queue.Full:
                    continue
            return False

        def _decode_thread():
            """Producer: decode audio and emit overlapping chunks."""
            try:
                container = av.open(long_source)
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
                        buf_parts = [remainder] if len(remainder) > 0 else []
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
                for _ in range(self.SENTINEL_RETRY_COUNT):
                    try:
                        chunk_queue.put(None, timeout=self.QUEUE_PUT_TIMEOUT)
                        break
                    except _queue.Full:
                        continue

        chunks_processed = 0
        best_result = None

        # Start pipeline
        decoder = threading.Thread(target=_decode_thread, daemon=True)
        decoder.start()

        early_stopped = False
        match_result = None

        while True:
            try:
                item = chunk_queue.get(timeout=self.QUEUE_GET_TIMEOUT)
            except _queue.Empty:
                # Decode thread may have died without sending sentinel
                if not decoder.is_alive():
                    break
                continue
            if item is None:
                break
            chunk_audio, sample_offset = item
            chunks_processed += 1
            result, _ = self._match_chunk(
                chunk_audio, sample_offset, sample_hashes, chunks_processed, verbose
            )
            if not verbose:
                print(f"\r[scan] Scanning chunk {chunks_processed} ...", end="", flush=True)

            best_count = best_result.match_count if best_result else -1
            if result is not None and result.match_count > best_count:
                best_result = result

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

        decoder.join(timeout=self.DECODER_JOIN_TIMEOUT)

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

        if not verbose:
            print()
        elapsed = _time.monotonic() - t_start
        if final["found"]:
            if verbose:
                print(f"[result] Match found at "
                      f"{_format_time(final['time_seconds'])} | "
                      f"count={final['match_count']} "
                      f"threshold={final['threshold']:.1f} | "
                      f"{chunks_processed} chunks | {elapsed:.1f}s")
            else:
                print(f"[result] Match found at "
                      f"{_format_time(final['time_seconds'])} | {elapsed:.1f}s")
        else:
            if verbose:
                print(f"[result] No match found | "
                      f"best count={final['match_count']} "
                      f"threshold={final['threshold']:.1f} | "
                      f"{chunks_processed} chunks | {elapsed:.1f}s")
            else:
                print(f"[result] No match found | {elapsed:.1f}s")

        return final
