# audiolocate

基於 Wang (2003) "An Industrial-Strength Audio Search Algorithm" 所提出的音訊指紋演算法的比對引擎。給定一段短音訊樣本與一段長參考音訊，判斷樣本是否出現在參考音訊中，並回傳精確的時間偏移量。

## 功能特色

- **頻譜峰值星座圖指紋** — 透過 STFT 頻譜局部峰值配對產生雜湊，實現高效比對
- **抗噪能力** — 在 3 dB SNR 條件下仍能正確辨識
- **長音訊快速定位** — 分塊串流搭配提早中斷機制，命中即停，無需掃描整段參考音訊，大幅縮短長音訊的比對時間
- **串流處理** — 支援大型音訊檔案的分塊串流比對，記憶體用量可控
- **平行解碼** — 音訊解碼與指紋計算同時進行，提升處理速度
- **多格式支援** — 透過 PyAV 支援 MP3、AAC、WAV 等常見音訊格式
- **多種輸入來源** — 接受檔案路徑、URL 及 file-like 物件

## 安裝

```bash
git clone https://github.com/JoeYang1412/audiolocate.git
cd audiolocate
pip install -r requirements.txt
```

## 快速開始

### 基本比對

```python
from audiolocate import AudioFingerprint

fp = AudioFingerprint()
result = fp.find_match("reference.wav", "sample.wav")

if result["found"]:
    print(f"Match found at {result['time_seconds']:.2f} seconds")
else:
    print("No match found")
```

### 串流比對（適用於大型檔案）

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

實測以 4 小時參考音訊定位 10 秒樣本：匹配點在開頭附近時，early-exit 於首批區塊即命中，約 **10 秒**完成；匹配點在最末尾需掃描所有區塊，約 **45 秒**完成（含網路 I/O、解碼與指紋計算）。

### 自訂參數

```python
fp = AudioFingerprint(
    sr=16000,                # Sample rate (default: 8000)
    n_fft=2048,              # FFT window size (default: 1024)
    peaks_per_second=50,     # Number of peaks per second (default: 30)
    significance_factor=2.5  # Statistical significance threshold factor (default: 3.0)
)
```

## 類別與方法

### `AudioFingerprint`

核心指紋比對類別。

| 方法 | 說明 |
|------|------|
| `load_audio(source)` | 載入音訊並重新取樣為單聲道 |
| `fingerprint_audio(audio)` | 產生音訊指紋（雜湊字典 + 幀數） |
| `find_match(reference, sample)` | 高階 API：比對兩段音訊，回傳結果字典 |
| `detect_peaks(spectrogram)` | 偵測頻譜圖中的局部峰值 |
| `generate_hashes(peaks)` | 從峰值星座圖產生配對雜湊 |
| `match(db_hashes, sample_hashes)` | 透過偏移直方圖比對兩組雜湊 |

### `StreamMatcher`

繼承自 `AudioFingerprint`，提供串流處理能力。

| 方法 | 說明 |
|------|------|
| `find_match_from_sources(long_source, short_source, ...)` | 串流比對主入口 |
| `fingerprint_source(source, chunk_seconds=300)` | 分塊串流建立指紋 |

### `MatchResult`

比對結果（NamedTuple）。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `is_match` | `bool` | 是否匹配 |
| `offset_frames` | `int` | 匹配位置的幀偏移量 |
| `match_count` | `int` | 匹配的雜湊數量 |
| `threshold` | `float` | 使用的自適應門檻值 |
| `noise_baseline` | `float` | 計算得到的噪音基準 |

## 主要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `sr` | 8000 | 取樣率 (Hz) |
| `n_fft` | 1024 | FFT 視窗大小 |
| `hop_length` | 256 | STFT 跳步長度 |
| `peak_neighborhood_size` | 15 | 峰值偵測鄰域大小 |
| `peaks_per_second` | 30 | 每秒目標峰值數量 |
| `fan_value` | 10 | 每個錨點的配對扇出數 |
| `target_t_min` / `target_t_max` | 2 / 100 | 配對時間差範圍（幀） |
| `target_f_min` / `target_f_max` | -30 / 60 | 配對頻率差範圍（bin） |
| `significance_factor` | 3.0 | 統計門檻倍數 |
| `prominence_factor` | 2.0 | 峰值突出度倍數 |
| `min_count_floor` | 5 | 最低匹配數量門檻 |


## 演算法簡介

1. **STFT** — 將音訊轉換為時頻頻譜圖
2. **峰值偵測** — 以局部最大值濾波器搭配區塊密度控制提取星座圖
3. **雜湊產生** — 將峰值配對編碼為 32 位元雜湊（頻率對 + 時間差）
4. **偏移直方圖** — 統計樣本與參考之間各時間偏移的雜湊命中數
5. **自適應門檻** — 結合最低門檻、統計顯著性與突出度三層判定

## References

> Wang, A. (2003). An Industrial-Strength Audio Search Algorithm.
> In *Proceedings of the 4th International Conference on Music
> Information Retrieval (ISMIR)*.

## 授權

請參閱專案根目錄的授權檔案。
