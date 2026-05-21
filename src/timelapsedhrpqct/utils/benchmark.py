from __future__ import annotations

import csv
import json
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from timelapsedhrpqct import __version__
from timelapsedhrpqct.dataset.layout import get_derivatives_root


@dataclass(slots=True)
class BenchmarkRecorder:
    """Collect lightweight wall-clock timing records for CLI runs."""

    dataset_root: Path
    command: str
    enabled: bool = False
    records: list[dict[str, object]] = field(default_factory=list)

    @contextmanager
    def section(self, name: str, **fields: object) -> Iterator[None]:
        """Record elapsed wall time for a named section when enabled."""
        if not self.enabled:
            yield
            return

        started = time.perf_counter()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - started
            record = {
                "command": self.command,
                "section": name,
                "status": status,
                "elapsed_s": round(elapsed, 6),
                **fields,
            }
            self.records.append(record)
            label = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
            label = f" {label}" if label else ""
            print(f"[benchmark] {name}{label}: {elapsed:.3f}s")

    def write(self) -> Path | None:
        """Write benchmark JSON and CSV summaries, returning the JSON path."""
        if not self.enabled:
            return None

        out_dir = get_derivatives_root(self.dataset_root) / "_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"benchmark_{self.command}_{stamp}"
        json_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"{stem}.csv"
        payload = {
            "version": 1,
            "created_utc": stamp,
            "command": self.command,
            "package_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "records": self.records,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        fieldnames = sorted({key for record in self.records for key in record.keys()})
        if fieldnames:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)
        else:
            csv_path.write_text("", encoding="utf-8")

        print(f"[benchmark] wrote: {json_path}")
        print(f"[benchmark] wrote: {csv_path}")
        return json_path


def benchmark_from_args(args, *, command: str, dataset_root: Path | None = None) -> BenchmarkRecorder:
    """Build a benchmark recorder from argparse args."""
    root = dataset_root or getattr(args, "dataset_root", None) or getattr(args, "output_root", None)
    if root is None:
        root = Path.cwd()
    return BenchmarkRecorder(
        dataset_root=Path(root),
        command=command,
        enabled=bool(getattr(args, "benchmark", False)),
    )
