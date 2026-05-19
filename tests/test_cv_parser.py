"""Extraction task — tests for the CV parser.

extract_cv_text() is pure (PDF / text extraction) — graded deterministically.
parse_cv() makes one LLM call — exercised with FakeLLM.
"""
import pytest

from core.cv_parser import MAX_CV_CHARS, extract_cv_text, parse_cv
from core.models import CVProfile, Role


class TestExtractCvText:
    def test_plain_text_bytes(self):
        assert extract_cv_text(b"Hello CV") == "Hello CV"

    def test_file_like_object(self):
        from io import BytesIO
        assert extract_cv_text(BytesIO(b"My CV text")) == "My CV text"

    def test_strips_surrounding_whitespace(self):
        assert extract_cv_text(b"  spaced CV  \n") == "spaced CV"

    def test_truncates_overlong_text(self):
        out = extract_cv_text(b"x" * (MAX_CV_CHARS + 5000))
        assert "truncated" in out
        assert len(out) < MAX_CV_CHARS + 200

    def test_rejects_unsupported_input_type(self):
        with pytest.raises(TypeError):
            extract_cv_text(12345)


class TestParseCv:
    def test_empty_text_returns_empty_profile_without_calling_llm(self, fake_llm):
        profile = parse_cv("", Role.DATA_ANALYST)
        assert isinstance(profile, CVProfile)
        assert profile.skills == []
        assert fake_llm.call_count == 0

    def test_whitespace_only_text_skips_the_llm(self, fake_llm):
        parse_cv("   \n  \t ", Role.DATA_ANALYST)
        assert fake_llm.call_count == 0

    def test_parses_via_one_llm_call(self, fake_llm):
        fake_llm.queue(CVProfile, CVProfile(skills=["SQL"], seniority="mid"))
        profile = parse_cv("10 years of SQL experience", Role.DATA_ANALYST)
        assert profile.skills == ["SQL"]
        assert fake_llm.call_count == 1
        assert fake_llm.calls[0].schema == "CVProfile"

    def test_cv_text_and_role_reach_the_prompt(self, fake_llm):
        fake_llm.queue(CVProfile, CVProfile())
        parse_cv("Senior Tableau analyst", Role.DATA_ANALYST)
        prompt = fake_llm.calls[0].user
        assert "Senior Tableau analyst" in prompt
        assert Role.DATA_ANALYST.value in prompt


class TestExtractCvFromPdf:
    def test_garbage_pdf_bytes_raise_valueerror(self):
        # Starts with the %PDF magic bytes but is not a real PDF.
        with pytest.raises(ValueError):
            extract_cv_text(b"%PDF-1.4 this is not actually a valid pdf file")

    def test_pdf_with_no_extractable_text_raises_valueerror(self):
        # A valid but blank PDF — pypdf opens it, but there is no text.
        from io import BytesIO
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = BytesIO()
        writer.write(buf)

        with pytest.raises(ValueError):
            extract_cv_text(buf.getvalue())
