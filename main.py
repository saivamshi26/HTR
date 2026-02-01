import os
import random
import argparse
import cv2
import numpy as np
import re
from path import Path
from typing import Tuple, List
from editdistance import eval as edit_distance

# ================= STABILITY =================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "0"

random.seed(0)
np.random.seed(0)
# =============================================

from dataloader_iam import Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


# ================= PATHS =====================
class FilePaths:
    fn_char_list = "../model/charList.txt"
    corpus_file = "../data/corpus.txt"
# =============================================


# ================= IMAGE SIZE =================
def get_img_height() -> int:
    return 32


def get_img_size() -> Tuple[int, int]:
    return 128, get_img_height()
# =============================================


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


# ================= CORPUS =====================
def load_corpus_words(path):
    words = set()
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for w in line.lower().split():
                if w.isalpha() and 2 <= len(w) <= 15:
                    words.add(w)
    return list(words)


CORPUS_WORDS = load_corpus_words(FilePaths.corpus_file)
# =============================================


# ================= LANGUAGE MODEL =================
def clean_word(w: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", w.lower())


def language_score(word: str) -> float:
    score = 0.0

    # reward common English letter patterns
    common_bigrams = ["th", "he", "er", "or", "ar", "rd", "on"]
    for bg in common_bigrams:
        if bg in word:
            score += 0.3

    # penalize uncommon endings
    bad_endings = ["h", "a", "e"]
    if len(word) > 4 and word[-1] in bad_endings:
        score -= 0.4

    # repeated letters
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:
            score -= 1.0

    # consonant runs
    vowels = "aeiou"
    run = 0
    for c in word:
        if c.isalpha() and c not in vowels:
            run += 1
            if run >= 4:
                score -= 1.5
        else:
            run = 0

    # length reward
    if 3 <= len(word) <= 8:
        score += 1.0

    return score


def language_correct(pred: str) -> str:
    pred = clean_word(pred)
    if not CORPUS_WORDS:
        return pred

    best_word = pred
    best_dist = 999

    for w in CORPUS_WORDS:
        d = edit_distance(pred, w)
        if d < best_dist:
            best_dist = d
            best_word = w
            if d == 0:
                break

    # SAFE correction only
    if best_dist <= 2:
        return best_word

    return pred
# =============================================


def print_candidates(scored, top_k=15):
    print(f"\nBEAM SEARCH CANDIDATES (top {top_k}):")
    for i, (w, s) in enumerate(scored[:top_k], 1):
        print(f"  {i:>2}. {w:<15} score={s:.2f}")


# ================= INFERENCE ==================
def infer(model: Model, fn_img: Path) -> None:
    img = cv2.imread(str(fn_img), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Image not found: {fn_img}"

    print(f"\nAnalyzing image: {fn_img}")

    # ---------- HANDWRITING NORMALIZATION ----------
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    # ----------------------------------------------

    os.makedirs("../debug_words", exist_ok=True)
    cv2.imwrite("../debug_words/word_1.png", img)

    preprocessor = Preprocessor(
        get_img_size(),
        dynamic_width=True,
        padding=48
    )

    proc = preprocessor.process_img(img)
    batch = Batch([proc], None, 1)

    # ---------- RAW OCR (BEST PATH) ----------
    raw_pred, _ = model.infer_batch(batch, True)
    raw_read = clean_word(raw_pred[0])

    print("\nRAW MODEL READ (after preprocessing):")
    print(" ", raw_read)

    # ---------- BEAM SEARCH ----------
    candidates, _ = model.infer_batch(batch, False)

    if not candidates:
        print("No OCR candidates.")
        return

    scored = []
    for w in candidates:
        w_clean = clean_word(w)
        if not w_clean:
            continue
        score = language_score(w_clean)
        scored.append((w_clean, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    print_candidates(scored, top_k=15)

    raw_word = scored[0][0]
    final_word = language_correct(raw_word)

    print("\n" + "=" * 40)
    print("FINAL RESULT (language-aware):")
    print(" ", final_word)
    print("=" * 40 + "\n")
# =============================================


# ================= MAIN ==================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_file",
        type=Path,
        required=True,
        help="Handwritten word image"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = Model(
        char_list_from_file(),
        decoder_type=DecoderType.BeamSearch,
        must_restore=True,
        dump=False
    )

    infer(model, args.img_file)


if __name__ == "__main__":
    main()
# =============================================
