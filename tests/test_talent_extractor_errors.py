from unittest.mock import patch

import pytest

from observatoire.talent_extractor_llm import (
    TalentExtractionUnavailable,
    extract_talent_profile,
)


def test_provider_failure_is_not_reported_as_a_rejection():
    with patch(
        "observatoire.talent_extractor_llm._call_openrouter_with_retry",
        side_effect=RuntimeError("OpenRouter unavailable"),
    ):
        with pytest.raises(TalentExtractionUnavailable):
            extract_talent_profile("Amina Benali", "Profil MRE valide avec activité en France.")


def test_empty_provider_response_is_retryable():
    with patch("observatoire.talent_extractor_llm._call_openrouter_with_retry", return_value=""):
        with pytest.raises(TalentExtractionUnavailable):
            extract_talent_profile("Amina Benali", "Profil MRE valide avec activité en France.")
