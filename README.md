# KYC Checker — Aadhaar & PAN (Gradio + Tesseract + spaCy)

Minimal offline KYC checker that OCRs uploaded Aadhaar & PAN images, extracts Name / DOB / ID numbers, compares the two documents, and reports a KYC pass/fail.

* UI: **Gradio** web app (`KYC.py`)
* OCR: **Tesseract** via `pytesseract`
* Name detection: **spaCy NER** (`en_core_web_md`) with simple heuristics & trimming to improve robustness
* ID/DOB extraction: simple **regex** (Aadhaar / PAN / DOB)
* Fuzzy name matching: **rapidfuzz** (optional) or Python `difflib` fallback

---

## Repository layout (important files)

```
.
├── Dockerfile                # supplied Dockerfile (build image)
├── run.sh (optional)         # example run script (you provided short run commands)
├── requirements.txt          # python deps
├── KYC.py                    # main Gradio app (the code we worked on)
├── test.py                   # test script (run first by your Docker CMD)
└── README.md                 # this file
```

---

## Quick start (build & run)

Use these exact commands:

```bash
# To build the Docker Image
docker build -t interiitps .

# To run the docker container
docker run -p 7860:7860 interiitps
```

After `docker run`, open the Gradio UI at:

```
http://localhost:7860
```

> Note: The app uses `demo.launch(..., share=True)` in code which attempts to create a public Gradio share link. That is optional — you can remove `share=True` if you don’t want the external share link.

---

## Dockerfile details (what it installs / runs)

Your Dockerfile (provided) does the following:

1. Uses `python:3.10` base image and sets `/app` as working directory.
2. Copies `requirements.txt` and installs Python packages (`pip install -r requirements.txt`).
3. Installs system Tesseract + tessdata packages:

   ```text
   apt-get update
   apt-get install -y tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-hin
   ```
4. Downloads spaCy medium model:

   ```bash
   python -m spacy download en_core_web_md
   ```
5. Copies the project code and runs:

   ```dockerfile
   CMD ["sh", "-c", "python -u ./test.py && python -u ./KYC.py"]
   ```

**Note:** to get live logs in Docker, make sure Python prints unbuffered output. If you see buffering, ensure `PYTHONUNBUFFERED=1` or use `python -u`.

---

## How the KYC approach works

1. **Image preprocessing** — the app rescales and autocontrasts uploaded images to improve OCR reliability. (Optional: install `opencv-python-headless` and enable denoise/adaptive thresholding for noisy mobile photos.)

2. **OCR** — `pytesseract.image_to_string(..., lang="eng+hin", config="--psm 6")` reads raw text from the image.

3. **Field extraction (regex)**:

   * **Aadhaar**: searches for `\d{4}\s?\d{4}\s?\d{4}` or `\d{12}` and formats as `xxxx xxxx xxxx`.
   * **PAN**: searches for the PAN pattern `[A-Z]{5}\d{4}[A-Z]` (case-insensitive).
   * **DOB**: finds `DD/MM/YYYY`, `DD-MM-YYYY`, or `DD MM YYYY` patterns.

4. **Name extraction (spaCy + heuristics)**:

   * Run spaCy NER (`en_core_web_md`) to find `PERSON` entities.
   * Use a hybrid scoring approach: prefer a line immediately above gender (`Male`/`Female`) or DOB lines; score spaCy candidates by proximity and shape
   * Filter header lines (e.g., `INCOME TAX`, `GOVT. OF INDIA`, `AADHAAR`) so they don't get chosen as names.

5. **Matching**:

   * Compare Aadhaar-name and PAN-name using fuzzy matching (`rapidfuzz` `token_sort_ratio` when available, otherwise `difflib`), normalized to lowercase and punctuation-removed.
   * Compare DOB strings after normalizing separators.
   * If both name similarity ≥ threshold (default **85%**) and DOB matches exactly (after normalization) → **KYC SUCCESSFUL**; otherwise return which fields mismatch and the extracted values.

---

## Requirements

Add (or verify) these in `requirements.txt`:

```
gradio>=3.0
pytesseract>=0.3.9
Pillow>=9.0
spacy>=3.4
rapidfuzz>=2.13      # optional but recommended
opencv-python-headless>=4.6   # optional; only if you want better preprocessing
```

Install spaCy model (Dockerfile already runs this):

```bash
python -m spacy download en_core_web_md
```

**Important:** Tesseract is a system binary and must be installed (Dockerfile already installs it). On Ubuntu outside Docker:

```bash
sudo apt update
sudo apt install -y tesseract-ocr
# (install language packages if needed)
sudo apt install -y tesseract-ocr-eng tesseract-ocr-hin
```

---

## Tuning & troubleshooting

* **Wrong name extraction / header chosen**

  * The code uses header keyword filtering and proximity heuristics; if spaCy still picks headers, add more header keywords to `HEADER_KEYWORDS` or increase the proximity window.
  * Trimming trailing lowercase tokens helps remove OCR artifacts like `Lemma,`. If you see lowercase tokens within the name (not only trailing), consider removing lowercase tokens in-line — but that risks false scrubbing for mixed-case names.

* **OCR quality low**

  * Try installing `opencv-python-headless` and enable the OpenCV preprocessing path in the script (denoising + adaptive threshold). This improves OCR on mobile photos.
  * Tweak Tesseract `--psm` (page segmentation mode) per document layout (`--psm 6`, `3`, or `11` can behave differently).

* **Name similarity threshold**

  * Default is `85%`. If you get false negatives (same person but OCR minor errors), lower threshold to 75–80. If false positives occur, raise threshold.

* **Model & memory**

  * `en_core_web_md` is chosen as a good middle ground for web apps: better than `sm`, much lighter than transformer models (`trf`). If you need higher accuracy and have compute, try `en_core_web_trf` (requires `spacy[transformers]` + `torch`).

* **Docker CMD behavior**

  * Current `CMD` runs `test.py` then `KYC.py`. If `test.py` never exits, `KYC.py` will not start. Use a small shell wrapper or run both scripts with background/`wait`, or run them in separate containers (recommended if they are separate services).

* **Gradio `share=True`**

  * `share=True` creates a public share link (if network access allowed). For local-only use, remove `share=True` to avoid public exposure.

---

## Development & debugging

* To debug candidate selection, add temporary prints/logs showing:

  * `ocr_raw` text
  * spaCy PERSON entities found
  * candidate scores and trimmed names
* To improve model performance on Indian names, consider fine-tuning a small spaCy NER with a few hundred labeled examples (fast to train).

---

## Security & privacy notes

* This tool extracts PII (names, DOB, ID numbers). In any real deployment:

  * Encrypt data at rest and in transit.
  * Keep logs redacted (do not store full raw OCR unless necessary).
  * Implement retention and deletion policies.
  * Add access controls and audit logs for reviewers.

---

## License
MIT License