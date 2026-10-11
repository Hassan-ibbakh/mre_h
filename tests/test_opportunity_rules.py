from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_engine import _has_expired_deadline


def test_expired_deadline_in_french_format_is_rejected():
    assert _has_expired_deadline("Date limite : 20 juillet 2026")


def test_expired_deadline_in_numeric_format_is_rejected():
    assert _has_expired_deadline("Dernier délai de dépôt : 20/07/2026")


def test_future_deadline_is_not_rejected():
    assert not _has_expired_deadline("Date limite de candidature : 31/12/2099")
