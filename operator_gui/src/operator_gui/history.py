"""Sorted bricks, appended to a CSV with one JPEG thumbnail each, so the tally survives restarts."""

import base64
import csv
import threading
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

GOOD, DEFECTIVE = "good", "defective"
CSV_NAME = "bricks.csv"
THUMBNAILS = "thumbnails"


@dataclass(frozen=True)
class BrickRecord:
    id: int
    time: str  # local time, ISO 8601
    verdict: str  # GOOD or DEFECTIVE: the dish the brick was sent to
    score: float | None  # anomaly score; None in a dry run
    looks: int | None  # views the decision used; None in a dry run
    thumbnail: str  # file name in the thumbnails folder, "" when there is none


class History:
    def __init__(self, folder: Path):
        self.folder = folder
        self.thumbnails = folder / THUMBNAILS
        self.thumbnails.mkdir(parents=True, exist_ok=True)
        self._csv = folder / CSV_NAME
        self._lock = threading.Lock()
        self._records = self._load()

    def _load(self) -> list[BrickRecord]:
        if not self._csv.exists():
            return []
        with self._csv.open(newline="") as file:
            return [
                BrickRecord(
                    id=int(row["id"]),
                    time=row["time"],
                    verdict=row["verdict"],
                    score=float(row["score"]) if row["score"] else None,
                    looks=int(row["looks"]) if row["looks"] else None,
                    thumbnail=row["thumbnail"],
                )
                for row in csv.DictReader(file)
            ]

    def add(self, verdict: str, score: float | None = None, looks: int | None = None,
            thumbnail_jpeg_b64: str | None = None) -> BrickRecord:
        with self._lock:
            brick_id = self._records[-1].id + 1 if self._records else 1
            thumbnail = ""
            if thumbnail_jpeg_b64:
                thumbnail = f"{brick_id:05d}.jpg"
                (self.thumbnails / thumbnail).write_bytes(base64.b64decode(thumbnail_jpeg_b64))
            record = BrickRecord(brick_id, datetime.now().isoformat(timespec="seconds"), verdict, score, looks,
                                 thumbnail)
            new_file = not self._csv.exists()
            with self._csv.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(BrickRecord)])
                if new_file:
                    writer.writeheader()
                writer.writerow(asdict(record))
            self._records.append(record)
            return record

    @property
    def records(self) -> list[BrickRecord]:
        with self._lock:
            return list(self._records)

    def counts(self) -> dict[str, int]:
        records = self.records
        defective = sum(r.verdict == DEFECTIVE for r in records)
        return {GOOD: len(records) - defective, DEFECTIVE: defective}
