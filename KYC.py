import re
import io
import sys
import subprocess
from difflib import SequenceMatcher
from io import BytesIO
from PIL import Image, ImageOps
import pytesseract
import gradio as gr
import spacy

print("Task 3:")

# try to load en_core_web_trf, download if missing
try:
    nlp = spacy.load("en_core_web_md")
except Exception:
    try:
        print("spaCy model en_core_web_md not found. Downloading...", file=sys.stderr, flush=True)
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"], check=True)
        nlp = spacy.load("en_core_web_md")
    except Exception as e:
        raise RuntimeError("Failed to download or load spaCy model en_core_web_md.") from e

# try to import rapidfuzz for better fuzzy matching; fallback to difflib
try:
    from rapidfuzz import fuzz
    _FUZZ_AVAILABLE = True
except Exception:
    _FUZZ_AVAILABLE = False


TESSERACT_LANG = "eng+hin"

# ---------------- image preprocessing ----------------
def preprocess_image(img: Image.Image) -> Image.Image:
    # Convert to RGB, autoscale/contrast and enlarge small images to help OCR.
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img)
    w, h = img.size
    max_dim = max(w, h)
    if max_dim < 1200:
        scale = 1200 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

# ---------------- OCR ----------------
def ocr_image(img: Image.Image) -> str:
    img = preprocess_image(img)
    # Using PSM 6 by default
    try:
        text = pytesseract.image_to_string(img, lang=TESSERACT_LANG, config="--psm 6")
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("Tesseract not found. Install Tesseract and ensure it's in PATH.")
    return text

# ---------------- extraction helpers ----------------
def normalize_whitespace(s: str) -> str:
    return " ".join([ln.strip() for ln in s.splitlines() if ln.strip()])

def extract_aadhaar(text: str) -> str:
    # aadhar format: 3 groups of 4 digits (total 12 digits)
    joined = normalize_whitespace(text)
    m = re.search(r"(\d{4}\s?\d{4}\s?\d{4})", joined)
    if not m:
        m = re.search(r"(\d{12})", joined)
    if m:
        digits = re.sub(r"\s+", "", m.group(0))
        if len(digits) == 12:
            return " ".join([digits[i:i+4] for i in range(0, 12, 4)])
        return m.group(0)
    return ""

def extract_pan(text: str) -> str:
    # PAN format: 5 letters, 4 digits, 1 letter
    joined = normalize_whitespace(text)
    m = re.search(r"([A-Z]{5}\d{4}[A-Z])", joined, flags=re.IGNORECASE)
    if m:
        return m.group(0).upper()
    return ""

def extract_dob(text: str) -> str:
    joined = normalize_whitespace(text)
    # dd/mm/yyyy or dd-mm-yyyy or dd mm yyyy
    m = re.search(r"((?:0[1-9]|[12][0-9]|3[01])[/\-\s](?:0[1-9]|1[0-2])[/\-\s](?:19|20)\d{2})", joined)
    if m:
        return m.group(1).replace('-', '/').strip()
    return ""

# ---------- header / heuristics (REPLACEMENT block) ----------
HEADER_KEYWORDS = [
    "india", "aadhaar", "of india", "government", "govt", "income tax",
    "department", "permanent account number", "signature", "pan", "aadhar",
    "आयकर", "भारत", "विभाग", "गवर्नमेंट"
]
GENDER_KEYWORDS = ["male", "female", "m", "f"]

def is_header_line(ln: str) -> bool:
    if not ln:
        return False
    low = ln.lower()
    for kw in HEADER_KEYWORDS:
        if kw in low:
            return True
    if len(ln.strip()) < 3:
        return True
    return False

def clean_line(ln: str) -> str:
    return ln.strip().replace("\u200c", "").replace("\u200d", "").strip()

def is_all_uppercase_name(s: str) -> bool:
    """
    Return True if the string contains at least one A-Z letter and contains NO a-z letters.
    Non-ASCII scripts will not match A-Z; this function enforces ASCII uppercase names.
    """
    if not s:
        return False
    # If any lowercase ascii letter exists, reject
    if re.search(r"[a-z]", s):
        return False
    # Require at least one uppercase ascii letter (avoid lines with only punctuation/numbers)
    return bool(re.search(r"[A-Z]", s))

def trim_lowercase_suffix(name: str) -> str:
    """
    Remove trailing tokens that contain lowercase ASCII letters or are mostly punctuation/digits.
    Keeps the initial uppercase tokens intact.
    Example: "SAMARTH SHARMA Lemma," -> "SAMARTH SHARMA"
    """
    if not name:
        return name
    toks = [t.strip() for t in re.split(r"\s+", name.strip()) if t.strip()]
    def token_is_lowercase_like(tok):
        # has any ascii lowercase -> mark bad
        if re.search(r"[a-z]", tok):
            return True
        # tokens with no alphabetic characters (punct/digits) are bad
        alpha_chars = re.findall(r"[A-Za-z\u0900-\u097F]", tok)
        if not alpha_chars:
            return True
        return False
    while toks and token_is_lowercase_like(toks[-1]):
        toks.pop()
    # strip punctuation from ends of tokens
    cleaned = [re.sub(r"^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$", "", t) for t in toks]
    cleaned = [t for t in cleaned if t]
    return " ".join(cleaned)

def find_candidate_by_proximity(lines):
    """
    Find the name line by looking for a line immediately before a gender or DOB line.
    We allow candidates that may have trailing lowercase tokens and will trim them.
    """
    for i, ln in enumerate(lines):
        llow = ln.lower()
        if any(g in llow for g in GENDER_KEYWORDS):
            for j in range(i-1, max(-1, i-6), -1):  # look up to 5 lines above
                cand_raw = clean_line(lines[j])
                if not cand_raw:
                    continue
                cand = trim_lowercase_suffix(cand_raw)
                if not cand:
                    continue
                if is_header_line(cand):
                    continue
                if re.search(r"\d", cand):   # skip lines with digits
                    continue
                if 1 < len(cand.split()) <= 6:
                    return cand
    # search for DOB lines and take the line before DOB (trim lowercase suffix)
    dob_re = re.compile(r"(?:0[1-9]|[12][0-9]|3[01])[/\-\s](?:0[1-9]|1[0-2])[/\-\s](?:19|20)\d{2}")
    for i, ln in enumerate(lines):
        if dob_re.search(ln):
            for j in range(i-1, max(-1, i-6), -1):
                cand_raw = clean_line(lines[j])
                if not cand_raw:
                    continue
                cand = trim_lowercase_suffix(cand_raw)
                if not cand:
                    continue
                if is_header_line(cand):
                    continue
                if re.search(r"\d", cand):
                    continue
                if 1 < len(cand.split()) <= 6:
                    return cand
    return None

def score_candidate_line(line: str, lines: list, idx: int):
    """
    Heuristic scoring with uppercase-only preference (but candidates may be trimmed).
    """
    if not line:
        return -999
    ln = line.strip()
    if is_header_line(ln):
        return -999
    if re.search(r"\d", ln):
        return -999

    s = 0
    # uppercase bonus if truly uppercase
    if ln == ln.upper():
        s += 30
    tokc = len(ln.split())
    if 2 <= tokc <= 4:
        s += 25
    elif tokc == 1:
        s += 5

    window = lines[max(0, idx-3): min(len(lines), idx+4)]
    window_text = " ".join(window).lower()
    if any(g in window_text for g in GENDER_KEYWORDS):
        s += 20
    dob_re = re.compile(r"(?:0[1-9]|[12][0-9]|3[01])[/\-\s](?:0[1-9]|1[0-2])[/\-\s](?:19|20)\d{2}")
    if dob_re.search(window_text):
        s += 15

    if 6 <= len(ln) <= 60:
        s += 5

    return s

def extract_name_spacy(text: str) -> str:
    """
    Hybrid method:
      1) split OCR into lines and de-header
      2) try proximity heuristic: line above gender or DOB (trim lowercase suffix)
      3) collect spaCy PERSON entities, filter headers, trim trailing lowercase, score each candidate
      4) choose best-scoring candidate; fallback to uppercase-line heuristic or first valid trimmed line
    """
    # split preserving order
    raw_lines = [ln for ln in text.splitlines()]
    lines = [clean_line(ln) for ln in raw_lines if ln.strip()]

    # 1) proximity heuristic
    prox = find_candidate_by_proximity(lines)
    if prox:
        return prox

    # 2) spaCy PERSON candidates (filtered then trimmed)
    doc = nlp(text)
    candidates = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            cand_raw = clean_line(ent.text)
            if not cand_raw:
                continue
            cand = trim_lowercase_suffix(cand_raw)
            if not cand:
                continue
            if is_header_line(cand):
                continue
            if re.search(r"\d", cand):
                continue
            # try to find where this candidate occurs in the original lines to compute proximity
            idxs = [i for i, L in enumerate(lines) if cand in L or cand.lower() in L.lower()]
            idx = idxs[0] if idxs else 0
            score = score_candidate_line(cand, lines, idx)
            candidates.append((cand, score))

    if candidates:
        # pick best by score (and break ties by longest)
        candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        best = candidates[0][0]
        if candidates[0][1] > 0:
            return best

    # 3) uppercase-line heuristic skipping header lines (trim lowercase suffix)
    for ln_raw in lines:
        ln = trim_lowercase_suffix(ln_raw)
        if len(ln) > 3 and ln.upper() == ln and not is_header_line(ln) and not re.search(r"\d", ln):
            return ln

    # 4) first long-ish non-digit trimmed line
    for ln_raw in lines:
        ln = trim_lowercase_suffix(ln_raw)
        if len(ln) > 3 and not is_header_line(ln) and not re.search(r"\d", ln):
            return ln

    # nothing found
    return ""

# ---------------- comparison helpers ----------------
def normalize_name(s: str) -> str:
    # lowercase, remove punctuation, extra spaces
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def name_similarity(a: str, b: str) -> float:
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    if not a_norm or not b_norm:
        return 0.0
    if _FUZZ_AVAILABLE:
        # token sort ratio is good for name permutations
        return fuzz.token_sort_ratio(a_norm, b_norm)
    else:
        # fallback to difflib ratio (0..1 -> convert to percentage)
        return SequenceMatcher(None, a_norm, b_norm).ratio() * 100.0

def dob_matches(dob1: str, dob2: str) -> bool:
    # Normalize separators to '/' and strip spaces
    if not dob1 or not dob2:
        return False
    d1 = re.sub(r"[\-\.\s]", "/", dob1).strip()
    d2 = re.sub(r"[\-\.\s]", "/", dob2).strip()
    # try to normalize common variants (DD/MM/YYYY)
    return d1 == d2

def process_both(aadhaar_img, pan_img):
    # Validate inputs
    if aadhaar_img is None or pan_img is None:
        return {
            "aadhaar_display": None,
            "pan_display": None,
            "aadhaar_raw": "",
            "pan_raw": "",
            "aadhaar_name": "",
            "aadhaar_dob": "",
            "aadhaar_num": "",
            "pan_name": "",
            "pan_dob": "",
            "pan_num": "",
            "result": "Please upload both Aadhaar and PAN images."
        }

    try:
        # OCR both
        aadhaar_pil = aadhaar_img if isinstance(aadhaar_img, Image.Image) else Image.open(BytesIO(aadhaar_img.read()))
        pan_pil = pan_img if isinstance(pan_img, Image.Image) else Image.open(BytesIO(pan_img.read()))

        aadhaar_raw = ocr_image(aadhaar_pil)
        pan_raw = ocr_image(pan_pil)

        # extract fields
        aadhaar_name = extract_name_spacy(aadhaar_raw)
        aadhaar_dob = extract_dob(aadhaar_raw)
        aadhaar_num = extract_aadhaar(aadhaar_raw)

        pan_name = extract_name_spacy(pan_raw)
        pan_dob = extract_dob(pan_raw)
        pan_num = extract_pan(pan_raw)

        # compare
        name_sim = name_similarity(aadhaar_name or "", pan_name or "")
        name_ok = name_sim >= 85  # threshold; tune as needed
        dob_ok = dob_matches(aadhaar_dob or "", pan_dob or "")

        # build result message
        if name_ok and dob_ok:
            result = f"KYC SUCCESSFUL\nName similarity: {name_sim:.1f}% (threshold 85%)\nDOB match: Yes"
        else:
            reasons = []
            if not name_ok:
                reasons.append(f"Name mismatch (similarity {name_sim:.1f}%)\n  Aadhaar name: {aadhaar_name or '<not found>'}\n  PAN name: {pan_name or '<not found>'}")
            if not dob_ok:
                reasons.append(f"DOB mismatch\n  Aadhaar DOB: {aadhaar_dob or '<not found>'}\n  PAN DOB: {pan_dob or '<not found>'}")
            result = "KYC FAILED\n" + "\n\n".join(reasons)

        return {
            "aadhaar_display": aadhaar_pil,
            "pan_display": pan_pil,
            "aadhaar_raw": aadhaar_raw,
            "pan_raw": pan_raw,
            "aadhaar_name": aadhaar_name,
            "aadhaar_dob": aadhaar_dob,
            "aadhaar_num": aadhaar_num,
            "pan_name": pan_name,
            "pan_dob": pan_dob,
            "pan_num": pan_num,
            "result": result
        }
    except Exception as e:
        return {
            "aadhaar_display": None,
            "pan_display": None,
            "aadhaar_raw": "",
            "pan_raw": "",
            "aadhaar_name": "",
            "aadhaar_dob": "",
            "aadhaar_num": "",
            "pan_name": "",
            "pan_dob": "",
            "pan_num": "",
            "result": f"ERROR during processing: {str(e)}"
        }

# ---------------- Gradio UI ----------------
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("## KYC Checker — Aadhaar & PAN")
    gr.Markdown("Upload Aadhaar and PAN images. The app will OCR both, extract Name and DOB (spaCy for names, regex for DOB/IDs), and compare them.")

    with gr.Row():
        with gr.Column():
            aadhaar_in = gr.Image(label="Upload Aadhaar image", type="pil")
        with gr.Column():
            pan_in = gr.Image(label="Upload PAN image", type="pil")

    btn = gr.Button("Run KYC Check")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Aadhaar")
            aadhaar_out_img = gr.Image(label="Aadhaar (processed)", interactive=False)
            aadhaar_raw_box = gr.Textbox(label="Aadhaar — Raw OCR text", lines=8)
            aadhaar_name_box = gr.Textbox(label="Aadhaar — Name", lines=1)
            aadhaar_dob_box = gr.Textbox(label="Aadhaar — DOB", lines=1)
            aadhaar_num_box = gr.Textbox(label="Aadhaar — Aadhaar number", lines=1)
        with gr.Column():
            gr.Markdown("### PAN")
            pan_out_img = gr.Image(label="PAN (processed)", interactive=False)
            pan_raw_box = gr.Textbox(label="PAN — Raw OCR text", lines=8)
            pan_name_box = gr.Textbox(label="PAN — Name", lines=1)
            pan_dob_box = gr.Textbox(label="PAN — DOB", lines=1)
            pan_num_box = gr.Textbox(label="PAN — PAN number", lines=1)

    result_box = gr.Textbox(label="KYC Result", interactive=False, lines=6)

    btn.click(
        fn=lambda a, p: (
            process_both(a, p)["aadhaar_display"],
            process_both(a, p)["pan_display"],
            process_both(a, p)["aadhaar_raw"],
            process_both(a, p)["pan_raw"],
            process_both(a, p)["aadhaar_name"],
            process_both(a, p)["aadhaar_dob"],
            process_both(a, p)["aadhaar_num"],
            process_both(a, p)["pan_name"],
            process_both(a, p)["pan_dob"],
            process_both(a, p)["pan_num"],
            process_both(a, p)["result"]
        ),
        inputs=[aadhaar_in, pan_in],
        outputs=[
            aadhaar_out_img, pan_out_img,
            aadhaar_raw_box, pan_raw_box,
            aadhaar_name_box, aadhaar_dob_box, aadhaar_num_box,
            pan_name_box, pan_dob_box, pan_num_box,
            result_box
        ],
    )

if __name__ == "__main__":
    print("Loading User Interface")
    print("Visit http://localhost:7860 after Gradio Launch\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)