"""CV Parser — extracts a structured profile from a user's uploaded CV.

Two public functions:
  extract_cv_text(file)  — pulls text out of a PDF or plain-text upload
  parse_cv(text, role)   — sends text to the LLM, returns a CVProfile

Usage in Streamlit:
    uploaded = st.file_uploader("Upload your CV", type=["pdf", "txt"])
    if uploaded:
        text = extract_cv_text(uploaded)
        profile = parse_cv(text, role=Role.DATA_ANALYST)
"""
from __future__ import annotations

import json
from io import BytesIO
from typing import Union

from pypdf import PdfReader

from core.llm import call_llm
from core.models import CVProfile, Role


# Cap the CV text we send to the LLM. Most CVs are well under 10K chars; this
# bound keeps prompts compact and avoids accidental token blow-ups on very
# long CVs (some people paste their entire LinkedIn export with recommendations).
MAX_CV_CHARS = 10_000


# ============================================================================
# Prompts
# ============================================================================

CV_PARSER_SYSTEM = """You are a CV parser.

Read the CV text below and extract a structured profile. Be conservative:

- skills: only include tools, technologies, frameworks, or languages
  explicitly mentioned (e.g. "PostgreSQL", "Tableau", "React"). Do NOT
  guess skills the candidate didn't write down.

- projects: 1-2 sentence descriptions of concrete projects the candidate
  built or led. Skip generic "team work" mentions.

- seniority: rough estimate based on years of experience or titles.
  Use one of: "entry", "mid", "senior".

- claimed_strengths: what the candidate emphasises in their summary,
  headline, or repeated themes.

- likely_gaps: skills commonly expected for the target role but MISSING
  from the CV. Be specific (e.g. "PowerBI", "Python") not vague
  (e.g. "more experience").

If the CV is sparse or unclear, return what you can — do not invent.

You MUST return JSON matching this exact schema (no prose, no code fences):
{schema}
"""

CV_PARSER_USER = """Target role: {role}

CV text:
\"\"\"
{cv_text}
\"\"\"

Extract the profile now."""


# ============================================================================
# Public API
# ============================================================================

def extract_cv_text(file: Union[bytes, BytesIO, "FileLike"]) -> str:
    """Extract text from a CV file. Supports PDF (.pdf) and plain text (.txt).

    Accepts:
      - raw bytes
      - a file-like object with .read() (e.g., Streamlit's UploadedFile)

    Returns: cleaned text content, capped at MAX_CV_CHARS.
    Raises:  ValueError on unparseable PDF or unsupported binary content.
    """
    data = _read_bytes(file)

    if data.startswith(b"%PDF"):
        text = _extract_from_pdf(data)
    else:
        # Best-effort decode for .txt / pasted text
        text = data.decode("utf-8", errors="replace")

    text = text.strip()
    if len(text) > MAX_CV_CHARS:
        text = text[:MAX_CV_CHARS] + "\n\n[...CV truncated to first 10,000 characters...]"
    return text


def parse_cv(cv_text: str, role: Role) -> CVProfile:
    """Parse a CV text into a structured CVProfile via one LLM call.

    If the text is empty, returns an empty CVProfile instead of failing.
    """
    if not cv_text or not cv_text.strip():
        return CVProfile()  # empty profile is valid and Pydantic-clean

    system = CV_PARSER_SYSTEM.format(
        schema=json.dumps(CVProfile.model_json_schema(), indent=2)
    )
    user = CV_PARSER_USER.format(role=role.value, cv_text=cv_text)

    return call_llm(
        system=system,
        user=user,
        schema=CVProfile,
        max_tokens=1024,
        temperature=0.0,
    )


# ============================================================================
# Internals
# ============================================================================

def _read_bytes(file) -> bytes:
    """Coerce a file-like object or raw input into bytes."""
    if isinstance(file, bytes):
        return file
    if hasattr(file, "read"):
        data = file.read()
        # Reset cursor in case the caller reads again (Streamlit re-runs).
        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
        if isinstance(data, str):
            return data.encode("utf-8")
        return data
    if isinstance(file, str):
        return file.encode("utf-8")
    raise TypeError(
        f"extract_cv_text expected bytes or file-like object, got {type(file).__name__}"
    )


def _extract_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf.

    Joins all pages with double newlines so section breaks survive.
    Pages that pypdf can't parse return empty string (handled by `or ''`).
    """
    try:
        reader = PdfReader(BytesIO(data))
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}") from e

    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""  # skip pages that fail (rare; usually images-only)
        if text.strip():
            pages.append(text)

    if not pages:
        raise ValueError(
            "PDF appears to have no extractable text. "
            "This usually means the PDF is a scanned image without an OCR layer. "
            "Try exporting your CV from Word, Google Docs, or LinkedIn instead."
        )

    return "\n\n".join(pages)
