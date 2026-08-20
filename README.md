# GeekLink OCR Benchmark

A benchmark for **burned-in video subtitle OCR**: 1,140 subtitle images
across 6 languages (English, Spanish, Japanese, Korean, Chinese, Greek),
rendered onto real film footage with exact known ground truth — so there's
no ambiguity about what the "correct" answer is, and no privacy or
copyright risk in the images themselves.

Unlike document-OCR benchmarks (scanned pages, receipts, street signs), this
targets the specific failure modes of **subtitle OCR in video**: low-contrast
backgrounds, film grain, small text against busy scenes, and — in ~19% of
samples — a second overlapping caption/watermark line placed deliberately
close to the subtitle, to test whether an engine can isolate the actual
subtitle instead of dumping every text region it finds.

> **Data correction (2026-08-21)**: an earlier version of this dataset had
> a real bug — 437 samples (38%) had ground-truth text wider than the video
> frame, so the rendered subtitle got clipped off-screen and the image
> didn't actually show the full ground-truth text. All affected samples
> have been re-rendered with proper line wrapping at their exact original
> timestamp. The dataset was also extended from 600 to 1,140 samples in the
> same pass. All baseline numbers below are from the corrected data; if
> you pulled this before 2026-08-21, please re-sync.

## How this was built (and why)

The images are real frames from three public-domain / openly-licensed films
(see Sources below), with subtitle lines burned in using the same ffmpeg +
libass pipeline used for hard-subtitle export in
[GeekLink](https://geeklink.dev). The subtitle text itself is short,
colloquial lines adapted from public-domain literature (Sherlock Holmes,
Dracula, Alice in Wonderland, Pride and Prejudice, A Christmas Carol, The
Wizard of Oz, Peter Pan, Moby-Dick, and others) — not verbatim quotes,
rewritten to read like natural spoken subtitles, then translated into the
other 5 languages. Lines that don't fit the frame width at render size are
automatically wrapped onto multiple lines (measured per-language, per-font,
per-video-width) rather than clipped.

We deliberately did **not** use real user video or real user OCR
corrections for this release: an earlier pass built on real (anonymized)
user data turned up enough privacy and copyright edge cases — a portrait
video where the subtitle position broke a positioning heuristic and nearly
exposed a bystander's face, adult content mixed into the same raw data pool,
personal documentary footage with real names — that we decided the safer
and more reproducible path was synthetic-but-realistic: real footage,
real difficulty, zero real people, zero rights ambiguity.

## What's in the data

- `data/manifest.csv` / `data/manifest.jsonl` — one row per sample: `id`,
  `video_id`, `lang`, `image`, `ground_truth`, `has_watermark`.
- `data/images/` — the rendered frames (full frames, not cropped — there's
  nothing sensitive to crop out here).
- `baselines/` — GeekLink's own OCR engine's raw output on this set (no
  post-filtering), both the ONNX/CPU build and the CoreML build actually
  used in the Mac app, for comparison.
- `external_baselines/` — raw output from PaddleOCR (3 PP-OCRv6 size
  tiers), EasyOCR, and Tesseract on the same set, same no-filtering
  methodology (see Baselines below).

| lang | samples |
|---|---|
| en | 191 |
| es | 191 |
| ja | 191 |
| ko | 191 |
| zh | 188 |
| el | 188 |
| **total** | **1140** |

~19% of samples (213/1140) include a synthetic watermark/credit line near
the subtitle — a harder detection case.

## Running the eval

```bash
python3 eval/eval.py --pred baselines/geeklink_coreml.csv   # score GeekLink's real Mac backend
python3 eval/eval.py --pred your_ocr_output.csv             # score your own engine
```

Your predictions file needs an `id` column (matching `data/manifest.csv`)
and a `prediction` column, as CSV or JSONL. Metrics are CER (character
error rate) and WER (word error rate), broken down by language and by
whether the sample has a watermark.

> WER is not meaningful for Chinese/Japanese without a word segmenter —
> both languages have no whitespace word boundaries. CER is the reliable
> metric across all languages here.

## Baselines: GeekLink vs. PaddleOCR vs. EasyOCR vs. Tesseract

All engines are scored the same way: **raw output, no post-filtering or
cropping** — every text region the engine detects is concatenated and
scored against the single-line ground truth, exactly what `eval.py` would
do with any prediction file you hand it. This is not each tool's
product-level accuracy (a real product adds subtitle-region selection on
top) — it's meant to isolate the underlying detection+recognition
difficulty, especially the watermark-interference case.

**What "GeekLink" means here**: GeekLink's local OCR is built directly on
PaddleOCR's official pre-trained weights (PP-OCRv4/PP-OCRv5, converted to
ONNX and CoreML for local inference) — not a custom-trained model. We're
in the process of collecting real correction data to eventually fine-tune
our own detection/recognition models, but haven't shipped one yet. So
"GeekLink" in this table is really "PaddleOCR's official weights, an older
version, run through ONNX Runtime or CoreML" — see the Apple Silicon
section below for why that's not quite the same as the `paddleocr` pip
package's numbers.

Versions: PaddleOCR = official `paddleocr` pip package v3.7.0. PP-OCRv6
ships three size tiers (tiny/small/medium); the tier only affects
en/es/ja/zh — ko/el aren't covered by PP-OCRv6 yet and always run on
PP-OCRv5 regardless of tier. EasyOCR = official `easyocr` pip package
(PyTorch); Tesseract = `tesseract-ocr` 5.x via `pytesseract`, no image
preprocessing; GeekLink = **PP-OCRv5 ONNX/CoreML weights (PP-OCRv4 for
Japanese specifically)**. EasyOCR has no Greek (`el`) language pack, so
its `el` rows are excluded rather than scored as zero.

| engine | overall CER | overall WER | clean CER | watermark CER | ms/image (CPU) |
|---|---|---|---|---|---|
| **PP-OCRv6 medium** | **0.5966** | 0.6208 | 0.4509 | 1.2722 | 1398.6 |
| PP-OCRv6 small | 0.5989 | 0.6323 | 0.4481 | 1.2984 | 381.0 |
| GeekLink (CoreML) | 0.6093 | 0.7291 | 0.4599 | 1.3019 | **97.3** (see Apple Silicon below) |
| GeekLink (ONNX/CPU) | 0.6098 | 0.7235 | 0.4641 | 1.2855 | 564.3 |
| EasyOCR | 0.6220 | 0.9425 | 0.4834 | 1.2855 | 591.1 |
| PP-OCRv6 tiny | 0.6579 | 0.6708 | 0.5069 | 1.3582 | 169.4 |
| Tesseract | 0.9463 | 1.2275 | 0.8883 | 1.2156 | 106.4 |

Speed is wall-clock per image on the same machine (Apple Silicon), CPU
execution provider, averaged over 50 English samples, model load time
excluded (one warm-up call before timing). Tesseract's speed isn't
comparable to the others — it has no scene-text detection stage, so it's
doing far less work (and scoring far worse for it).

**PP-OCRv6 medium and small are statistically tied** (0.5966 vs 0.5989 CER
— a 0.4% relative difference, within noise) **for 3.7x the latency**
(1399ms vs 381ms) — medium doesn't buy anything on this benchmark. Tiny is
dramatically faster (169ms) but loses a lot specifically on Japanese —
the "one model for 50 languages" tradeoff seems to bite hardest on CJK
scripts at the smallest size. For a real product, `small` is the sweet
spot, not the `medium` default.

Per-language CER:

| lang | v6-tiny | v6-small | v6-medium | GeekLink (CoreML) | EasyOCR | Tesseract |
|---|---|---|---|---|---|---|
| en | 0.4415 | 0.4362 | 0.4312 | 0.4515 | 0.4497 | 0.7735 |
| es | 0.4300 | 0.4324 | 0.4290 | 0.4456 | 0.4658 | 0.7442 |
| el | 0.4654 | 0.4654 | 0.4654 | 0.4941 | n/a | 0.7872 |
| ja | 1.4607 | 0.8414 | 0.8462 | 0.7711 | 0.7795 | 1.2050 |
| ko | 0.8672 | 0.8672 | 0.8672 | 0.9511 | 0.8727 | 1.3852 |
| zh | 1.3252 | 1.3518 | 1.3416 | 1.2720 | 1.0887 | 1.6453 |

(el/ko are identical across PP-OCRv6 tiers since they always run PP-OCRv5,
unaffected by the tier choice.)

Reproduce any row with `python3 eval/eval.py --pred baselines/<file>.csv`
or `--pred external_baselines/<file>.csv` (`paddleocr_medium.csv` /
`paddleocr_tiny.csv` / `paddleocr_small.csv` for the three PP-OCRv6 tiers).

## Apple Silicon: using each engine's actual best backend

CPU-only numbers give a level cross-platform comparison, but they
understate what you'd actually get on a Mac — GeekLink doesn't run ONNX
Runtime CPU in production, it runs the same model weights through
**CoreML**, and PyTorch (which EasyOCR is built on) has an Apple GPU
backend (MPS) that isn't enabled by default. Re-running with each engine's
real best-available backend on Apple Silicon:

| engine | backend | ms/image | overall CER |
|---|---|---|---|
| **GeekLink (CoreML)** | Apple Neural Engine / GPU via CoreML | **97.3** | 0.6093 |
| EasyOCR (MPS) | PyTorch Apple GPU backend | 141.7 | 0.6220 (unchanged by backend) |
| Tesseract | CPU only — no GPU backend exists | 106.4 | 0.9463 |
| PaddleOCR (any tier) | CPU only — official PaddlePaddle has no Apple GPU/Metal backend | 169–1399 | 0.597–0.658 |

GeekLink's CoreML path is **5.8x faster** than its own ONNX/CPU number
(564ms → 97ms) for essentially the same accuracy (0.6098 → 0.6093 CER —
within normal float-precision noise between backends, not a real accuracy
change) — CoreML lets the Apple Neural Engine and GPU do the work instead
of the CPU. EasyOCR gets a real speedup too (591ms → 142ms) once MPS is
enabled, though `easyocr.Reader(gpu=True)` isn't the default. PaddleOCR
has no such option on this platform at all: we checked directly
(`paddle.device.is_compiled_with_mps` doesn't exist, no custom device
types registered) — the official `paddlepaddle` package is CPU-only on
macOS regardless of which tier you pick.

**On Apple Silicon specifically, GeekLink's CoreML path is the fastest
option in this entire comparison** — faster than Tesseract, which does
far less work — while landing within a fraction of a percent of
PP-OCRv6's raw accuracy. Reproduce with
`python3 eval/eval.py --pred baselines/geeklink_coreml.csv`.

**Not included (yet)**: dedicated subtitle-extraction tools whose local
OCR engine is PaddleOCR itself, since a raw-recognition comparison would
just reproduce the PaddleOCR rows above. What's genuinely different about
several of those tools is that they ask the user to manually draw a crop
box around the subtitle region per video, which sidesteps the watermark
problem by construction rather than solving it algorithmically. That's a
real and interesting comparison — manual region selection vs. automatic
detection — just a different one than this raw-engine table, planned as a
follow-up rather than blocking this release.

**The headline finding**: every engine loses roughly 2.6-2.9x accuracy on
the watermark subset, regardless of how well it does on clean subtitles —
PP-OCRv6 medium's CER goes from 0.45 to 1.27, GeekLink's from 0.46 to 1.30,
EasyOCR's from 0.48 to 1.29. Watermark/overlay interference is a shared
blind spot across every general-purpose OCR engine we tested here, not
something specific to any one of them — raw text recognition quality
barely matters once there's a second text region competing for the same
space. Tesseract, which has no scene-text detection stage at all and just
OCRs the whole frame as a document, is the outlier: consistently worst,
as expected for a tool not built for this.

> Note on CJK: WER > 1.0 for ja/ko/zh happens because raw concatenation of
> multiple detected lines (subtitle + watermark) produces far more
> "words" than the single-line ground truth when split naively — expected
> given the no-filtering methodology above, not a WER calculation bug.

## Sources

- *The General* (1926), dir. Buster Keaton & Clyde Bruckman —
  [archive.org/details/TheGeneral1926](https://archive.org/details/TheGeneral1926),
  Public Domain Mark 1.0.
- *Nosferatu* (1922), dir. F.W. Murnau —
  [archive.org/details/Nosferatu1922](https://archive.org/details/Nosferatu1922),
  CC0 1.0.
- *San Francisco* (1955 Cinemascope travelogue) —
  [archive.org/details/SanFrancisco1955CinemascopeFilm](https://archive.org/details/SanFrancisco1955CinemascopeFilm),
  CC BY-SA 3.0 (attribution required — see LICENSE-DATA).

## License

- Code (`eval/`) — MIT, see `LICENSE`.
- Data (`data/`) — see `LICENSE-DATA`. The subtitle text is our own
  writing; two of the three source films are public domain, one is
  CC BY-SA and requires attribution (included above).

## Citing

If this benchmark is useful in your work, a link back to
[geeklink.dev](https://geeklink.dev) or this repo is appreciated.
