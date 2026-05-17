"""
voice_handler.py — Voice Interaction Layer for QuickCommerce Analytics.

Pipeline: Deepgram STT → NLP Intent Parser → Data Fetch → Scoring → TTS Response
Supports 5 intents: recommend, find_cheapest, check_delivery, trending, compare
"""

import re
import os
import sys
import io
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PRODUCT_NAMES, DEEPGRAM_API_KEY


# ─────────────────────────────────────────────
# INTENT PATTERNS (regex + keyword matching)
# ─────────────────────────────────────────────

INTENT_PATTERNS = {
    "find_cheapest": [
        r"cheap(?:est)?",
        r"lowest?\s*price",
        r"best\s*(?:price|deal|offer)",
        r"where.*cheap",
        r"affordable",
        r"budget",
    ],
    "check_delivery": [
        r"fast(?:est)?",
        r"quick(?:est)?",
        r"deliver(?:y)?\s*time",
        r"eta",
        r"how\s*(?:fast|quick|long)",
        r"speed",
        r"soonest",
    ],
    "trending": [
        r"trend(?:ing)?",
        r"popular",
        r"hot\s*(?:products?|items?)?",
        r"what.*(?:popular|trending|hot)",
        r"top\s*(?:products?|items?|selling)",
    ],
    "compare": [
        r"compar(?:e|ison)",
        r"(?:blinkit|zepto|swiggy).*(?:vs|versus|and|or).*(?:blinkit|zepto|swiggy)",
        r"side\s*by\s*side",
        r"difference\s*between",
    ],
    "recommend": [
        r"recommend",
        r"suggest",
        r"which\s*(?:platform|app|one)",
        r"where\s*(?:should|can|to)\s*(?:i|we)",
        r"best\s*(?:platform|app|option|place)",
        r"order\s*from",
    ],
}


def parse_intent(text):
    """
    Parse the user's query to determine intent.
    Returns one of: recommend, find_cheapest, check_delivery, trending, compare
    Default: recommend
    """
    text_lower = text.lower().strip()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent

    return "recommend"  # default intent


def extract_product(text):
    """
    Extract product name from query text.
    Matches against the 51-product catalogue.
    Returns (product_name, pack_size) or (None, None).
    """
    text_lower = text.lower().strip()

    # Try to match product names (longest first to avoid partial matches)
    sorted_products = sorted(PRODUCT_NAMES, key=len, reverse=True)

    matched_product = None
    for product in sorted_products:
        if product.lower() in text_lower:
            matched_product = product
            break

    # Try to extract pack size
    pack_size = None
    size_match = re.search(r'(\d+)\s*(?:ml|g|gm|gram|liter|litre|kg|pcs|tabs|pieces)', text_lower)
    if size_match:
        pack_size = size_match.group(0)

    return matched_product, pack_size


def extract_platforms(text):
    """Extract specific platform names mentioned in the query."""
    text_lower = text.lower()
    platforms = []
    if "blinkit" in text_lower:
        platforms.append("blinkit")
    if "zepto" in text_lower:
        platforms.append("zepto")
    if "swiggy" in text_lower or "instamart" in text_lower:
        platforms.append("swiggy")
    return platforms if platforms else None


def parse_query(text):
    """
    Full query parsing: intent + product + pack_size + platforms.
    Returns a dict with all parsed fields.
    """
    intent = parse_intent(text)
    product, pack_size = extract_product(text)
    platforms = extract_platforms(text)

    return {
        "raw_text": text,
        "intent": intent,
        "product": product,
        "pack_size": pack_size,
        "platforms": platforms,
    }


# ─────────────────────────────────────────────
# DEEPGRAM STT
# ─────────────────────────────────────────────

def transcribe_audio(audio_bytes, mimetype="audio/wav"):
    """
    Transcribe audio using Deepgram Nova-2 API.
    Returns (transcript_text, confidence) or (None, 0).
    """
    if not DEEPGRAM_API_KEY:
        print("Warning: DEEPGRAM_API_KEY not set. Using text fallback.")
        return None, 0.0

    try:
        from deepgram import DeepgramClient

        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes, "mimetype": mimetype},
            {
                "model": "nova-2",
                "language": "en-IN",
                "smart_format": True,
            }
        )

        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        confidence = response["results"]["channels"][0]["alternatives"][0]["confidence"]

        return transcript, confidence

    except Exception as e:
        print(f"Deepgram STT error: {e}")
        return None, 0.0


# ─────────────────────────────────────────────
# TTS (gTTS or pyttsx3)
# ─────────────────────────────────────────────

def text_to_speech(text):
    """
    Convert text to speech audio bytes using gTTS.
    Returns audio bytes (mp3 format) or None.
    """
    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang="en", tld="co.in")
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()

    except Exception as e:
        print(f"TTS error: {e}")
        return None


# ─────────────────────────────────────────────
# EXAMPLE QUERIES (shown when confidence < 0.75)
# ─────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "Where should I order milk from?",
    "Which is cheapest for 500g paneer?",
    "Which platform is fastest right now?",
    "What's trending today?",
    "Compare Blinkit and Zepto for chips",
    "Show me the cheapest oil available",
    "Best platform for dairy products",
    "How fast can I get Maggi delivered?",
]
