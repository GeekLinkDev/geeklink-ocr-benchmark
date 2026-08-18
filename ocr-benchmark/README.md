# GeekLink OCR Benchmark

A benchmark for **burned-in video subtitle OCR**: 600 subtitle images across
6 languages (English, Spanish, Japanese, Korean, Chinese, Greek), rendered
onto real film footage with exact known ground truth — so there's no
ambiguity about what the "correct" answer is, and no privacy or copyright
risk in the images themselves.

Unlike document-OCR benchmarks (scanned pages, receipts, street signs), this
targets the specific failure modes of **subtitle OCR in video**: low-contrast
backgrounds, film grain, small text against busy scenes, and — in ~19% of
samples — a second overlapping caption/watermark line placed deliberately
close to the subtitle, to test whether an engine can isolate the actual
subtitle instead of dumping every text region it finds.

## How this was built (and why)

The images are real frames from three public-domain / openly-licensed films
(see Sources below), with subtitle lines burned in using the same ffmpeg +
libass pipeline used for hard-subtitle export in
[GeekLink](https://geeklink.dev). The subtitle text itself is short,
colloquial lines adapted from public-domain literature (Sherlock Holmes,
Dracula, Alice in Wonderland, Pride and Prejudice, A Christmas Carol, and
others) — not verbatim quotes, rewritten to read like natural spoken
subtitles, then translated into the other 5 languages.

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
- `baselines/geeklink.csv` — GeekLink's own OCR engine's raw output on this
  set (no post-filtering), for comparison.
- `external_baselines/` — raw output from PaddleOCR, EasyOCR, and Tesseract
  on the same set, same no-filtering methodology (see Baselines below).

| lang | samples |
|---|---|
| en | 101 |
| es | 101 |
| ja | 101 |
| ko | 101 |
| zh | 98 |
| el | 98 |
| **total** | **600** |

~19% of samples (115/600) include a synthetic watermark/credit line near
the subtitle — a harder detection case.

## Running the eval

```bash
python3 eval/eval.py --pred baselines/geeklink.csv   # score GeekLink's own baseline
python3 eval/eval.py --pred your_ocr_output.csv      # score your own engine
```

Your predictions file needs an `id` column (matching `data/manifest.csv`)
and a `prediction` column, as CSV or JSONL. Metrics are CER (character
error rate) and WER (word error rate), broken down by language and by
whether the sample has a watermark.

> WER is not meaningful for Chinese/Japanese without a word segmenter —
> both languages have no whitespace word boundaries. CER is the reliable
> metric across all languages here.

## Baselines: GeekLink vs. PaddleOCR vs. EasyOCR vs. Tesseract

All four are scored the same way: **raw output, no post-filtering or
cropping** — every text region the engine detects is concatenated and
scored against the single-line ground truth, exactly what `eval.py` would
do with any prediction file you hand it. This is not each tool's
product-level accuracy (a real product adds subtitle-region selection on
top) — it's meant to isolate the underlying detection+recognition
difficulty, especially the watermark-interference case.

**What "GeekLink" means here**: GeekLink's local OCR is built directly on
PaddleOCR's official pre-trained weights (PP-OCRv4/PP-OCRv5, converted to
ONNX for local inference) — not a custom-trained model. We're in the
process of collecting real correction data to eventually fine-tune our own
detection/recognition models, but haven't shipped one yet. So "GeekLink"
in this table is really "PaddleOCR's official weights, an older version,
run through ONNX Runtime" — see the explanation below for why that's not
quite the same as the `paddleocr` pip package's numbers.

Versions: PaddleOCR = official `paddleocr` pip package v3.7.0. PP-OCRv6
ships three size tiers (tiny/small/medium); the tier only affects
en/es/ja/zh — ko/el aren't covered by PP-OCRv6 yet and always run on
PP-OCRv5 regardless of tier. EasyOCR = official `easyocr` pip package
(PyTorch, CPU); Tesseract = `tesseract-ocr` 5.x via `pytesseract`, no
image preprocessing; GeekLink = **PP-OCRv5 ONNX weights (PP-OCRv4 for
Japanese specifically)**, run through ONNX Runtime rather than native
PaddlePaddle. All CPU, no GPU. EasyOCR has no Greek (`el`) language pack,
so its `el` rows are excluded rather than scored as zero.

| engine | overall CER | overall WER | clean CER | watermark CER | ms/image (CPU) |
|---|---|---|---|---|---|
| PP-OCRv6 tiny | 0.6875 | 0.6689 | 0.5416 | 1.3369 | **169.4** |
| **PP-OCRv6 small** | **0.6330** | 0.6279 | 0.4866 | 1.2849 | 381.0 |
| PP-OCRv6 medium | 0.6293 | 0.6140 | 0.4889 | 1.2546 | 1398.6 |
| GeekLink | 0.6460 | 0.7228 | 0.5086 | 1.2579 | 564.3 |
| EasyOCR | 0.6814 | 0.9749 | 0.5596 | 1.2645 | 591.1 |
| Tesseract | 0.9670 | 1.2426 | 0.9099 | 1.2215 | 106.4 |

Speed is wall-clock per image on the same machine (Apple Silicon, CPU
only, no GPU), averaged over the 101 English samples, model load time
excluded (one warm-up call before timing). Tesseract's speed isn't
comparable to the others — it has no scene-text detection stage, so it's
doing far less work (and scoring far worse for it).

**The tier picture matters more than the single "PaddleOCR" number
above**: PP-OCRv6 **small** is both more accurate *and* 1.5x faster than
what GeekLink currently ships (0.633 CER / 381ms vs. 0.646 CER / 564ms) —
it dominates on both axes, not a trade-off. Medium buys essentially no
extra accuracy over small (0.6293 vs 0.6330, within noise) for 3.7x the
latency, so on this benchmark medium isn't worth it. Tiny is dramatically
faster (169ms) but loses a lot specifically on Japanese (CER 1.497 vs
small's 0.880) — the "one model for 50 languages" tradeoff seems to bite
hardest on CJK scripts at the smallest size. For our own roadmap, this
says "evaluate upgrading to PP-OCRv6 small," not "medium is state of the
art so bigger is better."

Per-language CER:

| lang | v6-tiny | v6-small | v6-medium | GeekLink | EasyOCR | Tesseract |
|---|---|---|---|---|---|---|
| en | 0.4493 | 0.4469 | 0.4456 | 0.4611 | 0.4706 | 0.7839 |
| es | 0.4304 | 0.4350 | 0.4281 | 0.4561 | 0.4670 | 0.7546 |
| el | 0.3883 | 0.3883 | 0.3883 | 0.4184 | n/a | 0.6646 |
| ja | 1.4968 | 0.8801 | 0.8769 | 0.8090 | 0.8276 | 1.2987 |
| ko | 1.0354 | 1.0354 | 1.0354 | 1.1300 | 1.0470 | 1.6021 |
| zh | 1.6850 | 1.7468 | 1.7273 | 1.6605 | 1.3353 | 1.8366 |

(el/ko are identical across tiers since they always run PP-OCRv5,
unaffected by the PP-OCRv6 tier choice.)

Reproduce any row with `python3 eval/eval.py --pred baselines/geeklink.csv`
or `--pred external_baselines/<file>.csv` (`paddleocr.csv` is the medium
tier; `paddleocr_tiny.csv` / `paddleocr_small.csv` are the other two).

**Not included (yet)**: dedicated subtitle-extraction tools like
[VideOCR](https://github.com/timminator/VideOCR) (744★). VideOCR's local
engine is PaddleOCR itself, so a raw-recognition comparison would just
reproduce the PaddleOCR row above — the genuinely different thing about
VideOCR is that it asks the user to manually draw a crop box around the
subtitle region per video, which sidesteps the watermark problem by
construction rather than solving it algorithmically. That's a real and
interesting comparison (manual region selection vs. automatic detection),
just a different one than this raw-engine table, and it currently requires
Docker or Linux/Windows to run (no native macOS build) — planned as a
follow-up rather than blocking this release.

**The headline finding**: every engine loses 2-2.6x accuracy on the
watermark subset, regardless of how well it does on clean subtitles —
PaddleOCR's CER goes from 0.49 to 1.25, GeekLink's from 0.51 to 1.26,
EasyOCR's from 0.56 to 1.26. Watermark/overlay interference is a shared
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
