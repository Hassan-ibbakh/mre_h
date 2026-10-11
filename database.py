import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime, text, Index
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./instance/talents.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(200), nullable=False)
    full_name = Column(String(150), nullable=True)
    role = Column(String(30), nullable=False, default="user")
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ProfileCache(Base):
    """Cache des profils/URLs déjà traités pour éviter la réutilisation."""
    __tablename__ = "profile_cache"

    id = Column(Integer, primary_key=True)
    source_url = Column(String(500), nullable=False, unique=True, index=True)
    source_name = Column(String(50), nullable=False)
    source_identifier = Column(String(200), nullable=True)
    processing_status = Column(String(20), nullable=False, default="pending")
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_cache_source_created", "source_name", "created_at"),
        Index("ix_cache_source_identifier", "source_identifier"),
    )

    def should_skip(self) -> bool:
        return self.processing_status in ("approved", "rejected")


def classify_mre_review_status(confidence: float | str | None, country: str | None) -> str:
    normalized_country = (country or "").strip().lower()
    if normalized_country in {"maroc", "morocco"}:
        return "rejected"

    if confidence is None:
        return "rejected"

    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return "pending"

    if score <= 1:
        score *= 100

    if score >= 60:
        return "approved"
    if score >= 40:
        return "masked"
    return "rejected"


class Post(Base):
    __tablename__ = "post"

    id = Column(Integer, primary_key=True)
    title = Column(String(150), nullable=False)
    sector = Column(String(50), nullable=False)
    country = Column(String(100))
    expertise_tags = Column(String(200))
    years_experience = Column(Integer)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    review_status = Column(String(20), nullable=False, default="pending")
    confidence_score = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source_url = Column(String(500))
    source_name = Column(String(150))
    image_url = Column(String(500))
    auto_discovered = Column(Boolean, default=False)

    _DEFAULT_SECTOR_LABELS = {
        "tech": "Technologie", "health": "Santé", "education": "Éducation",
        "agriculture": "Agriculture", "industry": "Industrie", "finance": "Finance",
        "creative": "Créatif", "social": "Social", "other": "Autre",
    }

    def sector_label(self, labels: dict | None = None) -> str:
        labels = labels or self._DEFAULT_SECTOR_LABELS
        return labels.get(self.sector, self.sector)

    def tags_list(self) -> list[str]:
        if not self.expertise_tags:
            return []
        return [t.strip() for t in self.expertise_tags.split(",") if t.strip()]

    def initials(self) -> str:
        parts = self.title.split()
        letters = "".join(p[0] for p in parts[:2] if p)
        return letters.upper() or "?"


def init_db():
    os.makedirs("instance", exist_ok=True)
    Base.metadata.create_all(bind=engine)

    with engine.begin() as connection:
        post_info = connection.execute(text("PRAGMA table_info(post)")).fetchall()
        post_columns = {row[1] for row in post_info}
        if "review_status" not in post_columns:
            connection.execute(
                text("ALTER TABLE post ADD COLUMN review_status VARCHAR(20) NOT NULL DEFAULT 'pending'")
            )
        if "confidence_score" not in post_columns:
            connection.execute(text("ALTER TABLE post ADD COLUMN confidence_score INTEGER"))
        connection.execute(
            text("UPDATE post SET review_status = 'approved' "
                 "WHERE is_active = 1 AND review_status = 'pending'")
        )

        user_info = connection.execute(text("PRAGMA table_info(user)")).fetchall()
        user_columns = {row[1] for row in user_info}
        for column_name, ddl in {
            "full_name": "ALTER TABLE user ADD COLUMN full_name VARCHAR(150)",
            "role": "ALTER TABLE user ADD COLUMN role VARCHAR(30) NOT NULL DEFAULT 'user'",
            "status": "ALTER TABLE user ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending'",
            "updated_at": "ALTER TABLE user ADD COLUMN updated_at DATETIME",
        }.items():
            if column_name not in user_columns:
                connection.execute(text(ddl))

        connection.execute(
            text("UPDATE user SET role = 'user' WHERE role IS NULL OR role = ''")
        )
        connection.execute(
            text("UPDATE user SET status = 'pending' WHERE status IS NULL OR status = ''")
        )


def is_profile_cached(url: str, db: 'SessionLocal' = None) -> bool:
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False

    try:
        cached = db.query(ProfileCache).filter(ProfileCache.source_url == url).first()
        return cached is not None and cached.should_skip()
    finally:
        if should_close:
            db.close()


def get_cached_profile_status(url: str, db: 'SessionLocal' = None) -> str | None:
    """Comme is_profile_cached(), mais retourne le VRAI statut ('approved' ou
    'rejected') au lieu d'un simple bool -- utile pour logger precisement
    pourquoi un profil a ete saute, plutot que de supposer 'approved' a tort."""
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False

    try:
        cached = db.query(ProfileCache).filter(ProfileCache.source_url == url).first()
        if cached is not None and cached.should_skip():
            return cached.processing_status
        return None
    finally:
        if should_close:
            db.close()


def cache_profile(
    url: str,
    source_name: str,
    status: str = "pending",
    source_identifier: str = None,
    db: 'SessionLocal' = None
) -> ProfileCache:
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False

    try:
        cached = db.query(ProfileCache).filter(ProfileCache.source_url == url).first()
        if cached:
            cached.processing_status = status
            cached.last_seen = datetime.now(timezone.utc)
        else:
            cached = ProfileCache(
                source_url=url,
                source_name=source_name,
                source_identifier=source_identifier,
                processing_status=status,
            )
            db.add(cached)
        db.commit()
        return cached
    finally:
        if should_close:
            db.close()


def get_cached_profiles_by_source(
    source_name: str,
    status: str = None,
    days_back: int = None,
    db: 'SessionLocal' = None
) -> list[ProfileCache]:
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False

    try:
        q = db.query(ProfileCache).filter(ProfileCache.source_name == source_name)
        if status:
            q = q.filter(ProfileCache.processing_status == status)
        if days_back:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            q = q.filter(ProfileCache.last_seen >= cutoff)
        return q.all()
    finally:
        if should_close:
            db.close()


def cleanup_old_cache(days_retention: int = 30, db: 'SessionLocal' = None):
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False

    try:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_retention)
        deleted_count = db.query(ProfileCache).filter(
            ProfileCache.last_seen < cutoff,
            ProfileCache.processing_status.in_(("approved", "rejected"))
        ).delete()
        db.commit()
        return deleted_count
    finally:
        if should_close:
            db.close()


def create_default_admin_if_missing(default_email: str, default_password: str):
    import bcrypt
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == default_email).first()
        if admin:
            admin.role = "admin"
            admin.status = "active"
            admin.full_name = admin.full_name or "Administrateur"
            db.commit()
            return admin

        hashed = bcrypt.hashpw(default_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        admin = User(
            email=default_email,
            password_hash=hashed,
            full_name="Administrateur",
            role="admin",
            status="active",
        )
        db.add(admin)
        db.commit()
        return admin
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()