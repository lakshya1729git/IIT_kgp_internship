from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config import DATABASE_URL

Base = declarative_base()


class TrafficEvent(Base):
    """
    Stores a single disruption event extracted by the LLM from one article.

    Maps to Layer 1 output: structured event with severity σ and confidence κ.
    Enhanced with: location inference metadata, impact duration, date formatting,
    multi-factor confidence, transport relevance flag, coordinates, and
    HGNN-adjusted confidence/severity for training feedback.
    """
    __tablename__ = "traffic_events"

    id              = Column(Integer, primary_key=True, autoincrement=True)

    # ── Source metadata ───────────────────────────────────────────────────────
    source          = Column(String(60),  nullable=False)
    source_url      = Column(Text,        nullable=True)
    tomtom_url      = Column(Text,        nullable=True)   # TomTom map deep-link
    raw_text        = Column(Text,        nullable=False)
    fetched_at      = Column(DateTime,    default=datetime.utcnow)
    published_date  = Column(String(10),  nullable=True)   # DD/MM/YY

    # ── LLM-extracted fields ──────────────────────────────────────────────────
    event_type      = Column(String(50),  nullable=False, default="unknown")
    transport_relevant = Column(Boolean,  default=True)
    location        = Column(String(255), nullable=True)
    location_inferred = Column(Boolean,   default=False)
    location_source = Column(String(30),  nullable=True)   # direct/road_name/landmark/etc.
    road_name       = Column(String(255), nullable=True)
    reason          = Column(Text,        nullable=True)
    time_mentioned  = Column(String(100), nullable=True)
    is_future_event = Column(Boolean,     default=False)

    # ── Coordinates — critical for HGNN road-road spatial adjacency ───────────
    # Without these, build_graph_from_db() produces no road-road edges during
    # training, meaning the HGNN learns without any spatial graph structure.
    lat             = Column(Float,       nullable=True)   # WGS84 latitude
    lon             = Column(Float,       nullable=True)   # WGS84 longitude

    # ── Severity σ(t) and confidence κ(t) — core Layer 1 outputs ─────────────
    severity        = Column(String(10),  nullable=False, default="low")
    severity_score  = Column(Integer,     nullable=False, default=1)   # 2/5/10
    confidence      = Column(Float,       nullable=False, default=0.0) # κ ∈ [0,1]
    llm_confidence  = Column(Float,       nullable=True)               # raw LLM κ
    source_reliability = Column(Float,    nullable=True)               # source score

    # ── Human-verified ground truth — used for HGNN training ─────────────────
    # Set via the label_events.py / auto_label.py scripts.
    # When present, graph_builder.py uses this instead of the LLM severity
    # and the trainer gives these rows VERIFIED_LABEL_WEIGHT × higher loss.
    verified_severity = Column(String(10), nullable=True)  # "low"/"medium"/"high"

    # ── HGNN-adjusted outputs — stored for training feedback loop ─────────────
    # Comparing pre/post HGNN values over time lets us validate whether the
    # graph adjustments are moving in the right direction.
    hgnn_confidence   = Column(Float,      nullable=True)  # HGNN-adjusted κ
    hgnn_severity     = Column(String(10), nullable=True)  # HGNN-predicted severity
    hgnn_multiplier   = Column(Float,      nullable=True)  # road disruption prob used
    severity_corrected = Column(Boolean,   default=False)  # True if HGNN overrode LLM

    # ── Impact duration ───────────────────────────────────────────────────────
    start_time_display    = Column(String(20),  nullable=True)   # DD/MM/YY
    estimated_end_time    = Column(String(100), nullable=True)   # natural language or DD/MM/YY HH:MM
    impact_duration_mins  = Column(Integer,     nullable=True)   # minutes
    impact_duration_label = Column(String(50),  nullable=True)   # "2–4 hours"
    duration_source       = Column(String(20),  nullable=True)   # api_official/llm_estimated/rule_based

    def __repr__(self):
        return (
            f"<TrafficEvent id={self.id} type={self.event_type} "
            f"road={self.road_name} severity={self.severity} "
            f"confidence={self.confidence:.2f} "
            f"lat={self.lat} lon={self.lon} "
            f"duration={self.impact_duration_label}>"
        )


# ── Engine & session factory ──────────────────────────────────────────────────

engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    """
    Create all tables if they don't exist yet.
    Also runs a safe column migration for existing databases — adds any new
    columns (lat, lon, hgnn_confidence, etc.) without dropping existing data.
    """
    Base.metadata.create_all(engine)
    _migrate_add_columns()


def _migrate_add_columns():
    """
    Safely add new columns to an existing traffic_events table.
    Uses PRAGMA table_info to check existing columns before ALTER TABLE,
    so this is safe to call on both fresh and legacy databases.
    """
    new_columns = [
        ("lat",                "REAL"),
        ("lon",                "REAL"),
        ("hgnn_confidence",    "REAL"),
        ("hgnn_severity",      "VARCHAR(10)"),
        ("hgnn_multiplier",    "REAL"),
        ("severity_corrected", "BOOLEAN DEFAULT 0"),
        ("verified_severity",  "VARCHAR(10)"),
    ]
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            result   = conn.execute(text("PRAGMA table_info(traffic_events)"))
            existing = {row[1] for row in result}   # column names
            for col_name, col_type in new_columns:
                if col_name not in existing:
                    conn.execute(text(
                        f"ALTER TABLE traffic_events ADD COLUMN {col_name} {col_type}"
                    ))
                    print(f"  [DB] Migrated: added column '{col_name}' to traffic_events")
            conn.commit()
    except Exception as e:
        # Non-fatal — new installs use create_all above, migration only for upgrades
        print(f"  [DB] Migration warning (non-fatal): {e}")


def get_session():
    """Return a new database session."""
    return SessionLocal()
