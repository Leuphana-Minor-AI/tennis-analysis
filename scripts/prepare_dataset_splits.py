#!/usr/bin/env python3
"""
Prepare video-level dataset splits for pose and stroke pipelines.

This script keeps all frames from a source video in the same split and
creates annotation-ready directories for both pose keypoints and stroke clips.
"""

import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)


@dataclass(frozen=True)
class FrameRecord:
    video_name: str
    frame_file: str
    frame_index: int | None
    timestamp_sec: float | None
    source_frame_path: str


FRAME_PATTERN = re.compile(r"^(?P<video>.+?)_frame_(?P<index>\d+)_t(?P<timestamp>\d+(?:\.\d+)?)\.jpg$", re.IGNORECASE)


def load_frame_records(frames_dir: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for frame_path in sorted(frames_dir.glob("*.jpg")):
        match = FRAME_PATTERN.match(frame_path.name)
        if match:
            video_name = match.group("video")
            frame_index = int(match.group("index"))
            timestamp_sec = float(match.group("timestamp"))
        else:
            video_name = frame_path.stem.split("_frame_")[0]
            frame_index = None
            timestamp_sec = None

        records.append(
            FrameRecord(
                video_name=video_name,
                frame_file=frame_path.name,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                source_frame_path=str(frame_path),
            )
        )

    return records


def group_frames_by_video(records: list[FrameRecord]) -> dict[str, list[FrameRecord]]:
    grouped: dict[str, list[FrameRecord]] = {}
    for record in records:
        grouped.setdefault(record.video_name, []).append(record)

    for video_name in grouped:
        grouped[video_name].sort(
            key=lambda item: (
                item.frame_index if item.frame_index is not None else 10**12,
                item.frame_file,
            )
        )

    return grouped


def assign_video_splits(video_names: list[str], seed: int = 42) -> dict[str, str]:
    if not video_names:
        return {}

    rng = random.Random(seed)
    shuffled = list(video_names)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, round(total * 0.7))
    val_count = 0 if total < 3 else max(1, round(total * 0.15))

    if train_count + val_count >= total:
        val_count = max(0, total - train_count - 1)

    test_count = total - train_count - val_count

    train_videos = shuffled[:train_count]
    val_videos = shuffled[train_count:train_count + val_count]
    test_videos = shuffled[train_count + val_count:train_count + val_count + test_count]

    split_map: dict[str, str] = {}
    for video_name in train_videos:
        split_map[video_name] = "train"
    for video_name in val_videos:
        split_map[video_name] = "val"
    for video_name in test_videos:
        split_map[video_name] = "test"

    return split_map


def copy_frame(record: FrameRecord, split_dir: Path, video_name: str) -> str:
    destination_dir = split_dir / video_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_path = destination_dir / record.frame_file
    shutil.copy2(record.source_frame_path, destination_path)
    return str(destination_path)


def build_annotation_scaffold(root_dir: Path) -> None:
    for split_name in ("train", "val", "test"):
        for target_name in ("pose", "stroke"):
            (root_dir / split_name / target_name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def prepare_dataset_splits(
    frames_dir: str = "data/frames",
    output_dir: str = "data/splits",
    seed: int = 42,
) -> dict:
    frames_path = Path(frames_dir)
    output_path = Path(output_dir)

    if not frames_path.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_path}")

    records = load_frame_records(frames_path)
    if not records:
        raise ValueError(f"No frames found in {frames_path}")

    grouped = group_frames_by_video(records)
    split_map = assign_video_splits(sorted(grouped.keys()), seed=seed)

    build_annotation_scaffold(output_path)

    summary = {
        "frames_dir": str(frames_path),
        "output_dir": str(output_path),
        "seed": seed,
        "videos": {},
        "splits": {"train": [], "val": [], "test": []},
        "frame_count": 0,
    }

    for video_name, video_frames in grouped.items():
        split_name = split_map.get(video_name, "train")
        summary["splits"][split_name].append(video_name)

        split_frames_dir = output_path / split_name / "frames"
        split_frames_dir.mkdir(parents=True, exist_ok=True)

        copied_frames = []
        for record in video_frames:
            copied_frame_path = copy_frame(record, split_frames_dir, video_name)
            copied_frames.append(copied_frame_path)

        summary["videos"][video_name] = {
            "split": split_name,
            "frame_count": len(video_frames),
            "frames": [asdict(record) for record in video_frames],
        }
        summary["frame_count"] += len(video_frames)

        video_manifest = {
            "video_name": video_name,
            "split": split_name,
            "frame_count": len(video_frames),
            "frames": [asdict(record) for record in video_frames],
            "copied_frames": copied_frames,
        }
        write_json(output_path / split_name / f"{video_name}_manifest.json", video_manifest)

        pose_stub = {
            "video_name": video_name,
            "task": "pose",
            "split": split_name,
            "format": "frame_keypoints",
            "frames": [
                {
                    "frame_file": record.frame_file,
                    "frame_index": record.frame_index,
                    "timestamp_sec": record.timestamp_sec,
                    "keypoints": [],
                }
                for record in video_frames
            ],
        }
        stroke_stub = {
            "video_name": video_name,
            "task": "stroke",
            "split": split_name,
            "format": "clip_windows",
            "clips": [],
        }
        write_json(output_path / split_name / "pose" / f"{video_name}.json", pose_stub)
        write_json(output_path / split_name / "stroke" / f"{video_name}.json", stroke_stub)

    dataset_manifest = {
        "summary": summary,
        "split_map": split_map,
    }
    write_json(output_path / "dataset_manifest.json", dataset_manifest)

    return dataset_manifest


def main() -> None:
    try:
        dataset_manifest = prepare_dataset_splits()
    except Exception as exc:
        print(f"❌ Failed to prepare splits: {exc}")
        sys.exit(1)

    summary = dataset_manifest["summary"]
    print("✅ Dataset splits prepared")
    print(f"   Frames processed: {summary['frame_count']}")
    print(f"   Train videos: {len(summary['splits']['train'])}")
    print(f"   Val videos: {len(summary['splits']['val'])}")
    print(f"   Test videos: {len(summary['splits']['test'])}")
    print(f"   Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()