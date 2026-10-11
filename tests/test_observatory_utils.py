from observatoire.utils import tokenize_keywords
from database import Base, Post, classify_mre_review_status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_tokenize_keywords_splits_on_spaces_and_commas():
    assert tokenize_keywords("Data Science, Python   engineer") == ["data", "science", "python", "engineer"]


def test_mre_review_status_thresholds():
    assert classify_mre_review_status(60, "France") == "approved"
    assert classify_mre_review_status(59, "France") == "masked"
    assert classify_mre_review_status(40, "France") == "masked"
    assert classify_mre_review_status(39, "France") == "rejected"


def test_masked_profile_is_persisted_and_public_data_is_sanitized():
    from main import _public_profile_data

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    post = Post(
        title="Amina Benali",
        sector="tech",
        country="France",
        description="Private phone +33 1 23 45 67 89",
        source_name="Private source",
        source_url="https://private.example/profile",
        review_status="masked",
        confidence_score=59,
        is_active=True,
    )
    session.add(post)
    session.commit()

    saved = session.query(Post).one()
    public = _public_profile_data(saved, match_score=81)

    assert saved.review_status == "masked"
    assert saved.confidence_score == 59
    assert public["title"] == "Profil MRE masqué"
    assert public["description"] == "Les informations détaillées de ce profil sont masquées."
    assert public["source_name"] == ""
    assert public["source_url"] == ""
    assert public["match_score"] == 81
