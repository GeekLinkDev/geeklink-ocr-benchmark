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

## Baseline (GeekLink's own OCR, raw output — no post-filtering)

This baseline concatenates every text region the engine detects, exactly
like `eval.py` would score any other engine's raw dump — no subtitle
selection or watermark filtering applied. It's meant to show the *scale*
of the watermark problem, not GeekLink's product-level accuracy.

| lang | n | CER | WER |
|---|---|---|---|
| el | 98 | 0.4184 | 0.4677 |
| en | 101 | 0.4611 | 0.4199 |
| es | 101 | 0.4561 | 0.4292 |
| ja | 101 | 0.8090 | 1.9010 |
| ko | 101 | 1.1300 | 1.1900 |
| zh | 98 | 1.6605 | 3.4082 |
| **ALL** | **600** | **0.6460** | **0.7228** |

| | n | CER | WER |
|---|---|---|---|
| clean | 485 | 0.5086 | 0.6241 |
| **watermark** | 115 | **1.2579** | 1.1502 |

The watermark subset more than doubles CER (0.51 → 1.26) — an engine that
doesn't filter out nearby overlapping text pays for it, which is exactly
the failure mode this benchmark is designed to surface. Run
`python3 eval/eval.py --pred baselines/geeklink.csv` yourself to reproduce
this table, or substitute your own engine's output.

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
