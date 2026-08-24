#!/usr/bin/env python3
"""Load archive metadata and topic records for dataset alignment."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ArchiveLoadError(RuntimeError):
    pass


def load_manifest(archive_dir: Path) -> Dict[str, Any]:
    manifest_path = archive_dir / "manifest.json"
    if not manifest_path.exists():
        raise ArchiveLoadError(f"manifest.json not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _topic_dir_name(topic_name: str) -> str:
    sanitized = topic_name.strip("/")
    if not sanitized:
        return "root_topic"
    return sanitized.replace("/", "__")


def find_topic_entry(manifest: Dict[str, Any], topic_name: str) -> Optional[Dict[str, Any]]:
    for entry in manifest.get("topics", []):
        if entry.get("topic_name") == topic_name:
            return entry
    return None


def find_first_topic_entry(manifest: Dict[str, Any], candidates: Iterable[str]) -> Tuple[str, Dict[str, Any]]:
    for topic in candidates:
        entry = find_topic_entry(manifest, topic)
        if entry is not None:
            return topic, entry
    raise ArchiveLoadError(f"None of topics found in manifest: {list(candidates)}")


def topic_file_path(archive_dir: Path, topic_name: str, filename: str) -> Path:
    return archive_dir / "topics" / _topic_dir_name(topic_name) / filename


def read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise ArchiveLoadError(f"jsonl.gz not found: {path}")
    records: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def pick_time_ns(record: Dict[str, Any]) -> int:
    header_time = record.get("header_time_ns")
    if header_time is not None:
        return int(header_time)
    bag_time = record.get("bag_time_ns")
    if bag_time is not None:
        return int(bag_time)
    raise ArchiveLoadError("Record missing both header_time_ns and bag_time_ns")


def extract_joint_position(msg_record: Dict[str, Any]) -> List[float]:
    msg = msg_record.get("message", {})
    pos = msg.get("position")
    if pos is None:
        raise ArchiveLoadError("Joint message missing message.position")
    return [float(x) for x in pos]
