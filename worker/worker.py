import os
import time
from datetime import datetime, timezone
from typing import Tuple

from dotenv import load_dotenv
from supabase import create_client

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# -----------------------
# Env / config
# -----------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

HF_MODEL = os.getenv("HF_MODEL", "emngarcia/deberta_mh_benign_worrisome")
MODEL_VERSION = os.getenv("MODEL_VERSION", "hackathon-v1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "1.0"))
MAX_LEN = int(os.getenv("MAX_LEN", "256"))
MIN_WORDS = int(os.getenv("MIN_WORDS", "5"))

# -----------------------
# Clients
# -----------------------
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# -----------------------
# Load model (once)
# -----------------------
print(f"Loading model {HF_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
model.eval()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(f"Model loaded on {device}")


# -----------------------
# Inference
# -----------------------
def infer_one(text: str) -> Tuple[float, str]:
    enc = tokenizer(
        [text],
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
    score, idx = torch.max(probs[0], dim=-1)
    label = model.config.id2label.get(int(idx.item()), str(int(idx.item())))
    return float(score.item()), label


# -----------------------
# DB helpers
# -----------------------
def claim_events(batch_size: int):
    """Atomically claim a batch of queued events and mark them processing."""
    res = sb.rpc("claim_keyboard_events", {"batch_size": batch_size}).execute()
    return res.data or []


def word_count(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len(t.split())


def main() -> None:
    print(
        f"Worker started. model={HF_MODEL} device={device} "
        f"batch={BATCH_SIZE} min_words={MIN_WORDS}"
    )

    while True:
        events = claim_events(BATCH_SIZE)

        if not events:
            time.sleep(SLEEP_SECONDS)
            continue

        print(f"Claimed {len(events)} events")

        for ev in events:
            event_id = ev["id"]
            text = (ev.get("text") or "").strip()

            # Discard if too few words
            if word_count(text) < MIN_WORDS:
                sb.table("keyboard_events") \
                    .update({"status": "discarded"}) \
                    .eq("id", event_id) \
                    .execute()
                print(f"Discarded {event_id}: '{text[:40]}'")
                continue

            # Run inference
            try:
                score, label = infer_one(text)
                print(f"  {event_id}: label={label} score={score:.3f} text='{text[:60]}'")

                # Write to predictions
                sb.table("predictions").insert({
                    "event_id": event_id,
                    "label": label,
                    "score": score,
                    "model_version": MODEL_VERSION,
                    "input_text": text,
                }).execute()

                # Mark event done
                sb.table("keyboard_events") \
                    .update({"status": "done"}) \
                    .eq("id", event_id) \
                    .execute()

            except Exception as e:
                print(f"  FAILED {event_id}: {e}")
                sb.table("keyboard_events") \
                    .update({"status": "failed", "error": str(e)[:5000]}) \
                    .eq("id", event_id) \
                    .execute()

        time.sleep(0.2)


if __name__ == "__main__":
    main()