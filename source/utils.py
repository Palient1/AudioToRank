from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path


def parse_user_datetime(value: str, is_end: bool) -> datetime:
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        day = datetime.strptime(value, "%Y-%m-%d").date()
        return datetime.combine(day, time.max if is_end else time.min)

    cleaned = value.replace("Z", "+00:00")
    if len(cleaned) >= 5 and (cleaned[-5] in ["+", "-"] and cleaned[-3] != ":"):
        cleaned = cleaned[:-2] + ":" + cleaned[-2:]
    return datetime.fromisoformat(cleaned)


def parse_filename(path: Path) -> tuple[str, datetime, str] | None:
    stem = path.stem
    parts = stem.split("%")
    if len(parts) < 4:
        return None

    full_name = " ".join(parts[:3])
    dt_str = parts[3]

    try:
        cleaned = dt_str.replace("Z", "+00:00")
        if len(cleaned) >= 5 and (cleaned[-5] in ["+", "-"] and cleaned[-3] != ":"):
            cleaned = cleaned[:-2] + ":" + cleaned[-2:]
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None

    return full_name, dt, dt_str


def normalize_for_compare(value: datetime, reference: datetime) -> datetime:
    if value.tzinfo and reference.tzinfo:
        return value.astimezone(timezone.utc)
    if value.tzinfo and not reference.tzinfo:
        return value.replace(tzinfo=None)
    if not value.tzinfo and reference.tzinfo:
        return value
    return value


def should_process(file_dt: datetime, start_dt: datetime | None, end_dt: datetime | None) -> bool:
    if start_dt:
        if normalize_for_compare(file_dt, start_dt) < normalize_for_compare(start_dt, file_dt):
            return False
    if end_dt:
        if normalize_for_compare(file_dt, end_dt) > normalize_for_compare(end_dt, file_dt):
            return False
    return True
