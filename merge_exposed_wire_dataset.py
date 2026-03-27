from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OUTPUT_CLASS_ID = 0


@dataclass
class Sample:
    source_name: str
    image_path: Path
    label_path: Path
    image_ext: str
    remapped_label_lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SET1 and SET3 exposed-wire YOLO datasets into one cleaned dataset."
    )
    parser.add_argument(
        "--set1",
        type=Path,
        default=Path("Exposed Wire") / "SET1",
        help="Path to SET1 directory.",
    )
    parser.add_argument(
        "--set3",
        type=Path,
        default=Path("Exposed Wire") / "SET3",
        help="Path to SET3 directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset") / "Exposed_Wire",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--warn",
        action="store_true",
        help="Print warnings for skipped files.",
    )
    return parser.parse_args()


def iter_images(dataset_root: Path) -> Iterable[Path]:
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_label_for_image(image_path: Path) -> Path | None:
    # Typical YOLO structure keeps sibling folders named images/ and labels/.
    parts_lower = [part.lower() for part in image_path.parts]
    try:
        images_idx = parts_lower.index("images")
    except ValueError:
        return None

    label_parts = list(image_path.parts)
    label_parts[images_idx] = "labels"
    label_candidate = Path(*label_parts).with_suffix(".txt")
    return label_candidate


def remap_and_validate_label_lines(label_path: Path) -> tuple[list[str], str | None]:
    try:
        raw_lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [], f"cannot read label ({exc})"

    if not raw_lines:
        return [], "empty label file"

    cleaned_lines: list[str] = []
    for idx, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        tokens = stripped.split()
        if len(tokens) != 5:
            continue

        try:
            # Keep bbox values unchanged; only remap class id.
            float(tokens[1])
            float(tokens[2])
            float(tokens[3])
            float(tokens[4])
        except ValueError:
            continue

        cleaned_lines.append(
            f"{OUTPUT_CLASS_ID} {tokens[1]} {tokens[2]} {tokens[3]} {tokens[4]}"
        )

    if not cleaned_lines:
        return [], "no valid YOLO rows in label file"

    return cleaned_lines, None


def collect_valid_samples(dataset_root: Path, source_name: str, warn: bool) -> tuple[list[Sample], int]:
    valid_samples: list[Sample] = []
    skipped = 0

    for image_path in iter_images(dataset_root):
        label_path = find_label_for_image(image_path)
        if label_path is None:
            skipped += 1
            if warn:
                print(f"[WARN] Skipping {image_path}: not under an images/ directory")
            continue

        if not label_path.exists():
            skipped += 1
            if warn:
                print(f"[WARN] Skipping {image_path}: missing label {label_path}")
            continue

        remapped_lines, reason = remap_and_validate_label_lines(label_path)
        if reason is not None:
            skipped += 1
            if warn:
                print(f"[WARN] Skipping {image_path}: {reason}")
            continue

        valid_samples.append(
            Sample(
                source_name=source_name,
                image_path=image_path,
                label_path=label_path,
                image_ext=image_path.suffix.lower(),
                remapped_label_lines=remapped_lines,
            )
        )

    return valid_samples, skipped


def ensure_output_layout(output_root: Path) -> None:
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy split folders from older runs so only merged folders remain.
    for legacy_dir in (
        images_dir / "train",
        images_dir / "val",
        labels_dir / "train",
        labels_dir / "val",
    ):
        if legacy_dir.exists() and legacy_dir.is_dir():
            shutil.rmtree(legacy_dir)


def clear_previous_outputs(output_root: Path) -> None:
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"

    for image_file in images_dir.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
            image_file.unlink(missing_ok=True)

    for label_file in labels_dir.iterdir():
        if label_file.is_file() and label_file.suffix.lower() == ".txt":
            label_file.unlink(missing_ok=True)


def write_sample(
    sample: Sample,
    output_root: Path,
    destination_stem: str,
) -> bool:
    image_out = output_root / "images" / f"{destination_stem}{sample.image_ext}"
    label_out = output_root / "labels" / f"{destination_stem}.txt"

    try:
        shutil.copy2(sample.image_path, image_out)
        label_out.write_text("\n".join(sample.remapped_label_lines) + "\n", encoding="utf-8")
        return True
    except OSError as exc:
        print(f"[WARN] Failed to write {destination_stem}: {exc}")
        # Best effort cleanup to keep image/label pairs consistent.
        if image_out.exists():
            image_out.unlink(missing_ok=True)
        if label_out.exists():
            label_out.unlink(missing_ok=True)
        return False


def write_data_yaml(output_root: Path) -> None:
    yaml_content = "\n".join(
        [
            "path: .",
            "train: images",
            "val: images",
            "",
            "nc: 1",
            "names:",
            "  0: exposed_wire",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(yaml_content, encoding="utf-8")


def generate_destination_name(counters: dict[str, int], source_name: str) -> str:
    counters["merged"] = counters.get("merged", 0) + 1
    return f"exposed_wire_{counters['merged']:06d}"


def main() -> None:
    args = parse_args()

    set1_root = args.set1.resolve()
    set3_root = args.set3.resolve()
    output_root = args.output.resolve()

    if not set1_root.exists() or not set1_root.is_dir():
        raise SystemExit(f"SET1 path is invalid: {set1_root}")
    if not set3_root.exists() or not set3_root.is_dir():
        raise SystemExit(f"SET3 path is invalid: {set3_root}")

    ensure_output_layout(output_root)
    clear_previous_outputs(output_root)

    set1_samples, skipped_set1 = collect_valid_samples(set1_root, "set1", args.warn)
    set3_samples, skipped_set3 = collect_valid_samples(set3_root, "set3", args.warn)

    all_samples = set1_samples + set3_samples
    total_skipped = skipped_set1 + skipped_set3

    written = 0
    failed_writes = 0
    name_counters: dict[str, int] = {"merged": 0}

    for sample in all_samples:
        destination_stem = generate_destination_name(name_counters, sample.source_name)
        if write_sample(sample, output_root, destination_stem):
            written += 1
        else:
            failed_writes += 1

    write_data_yaml(output_root)

    print(f"Total valid images processed: {written}")
    print(f"Total skipped: {total_skipped + failed_writes}")
    print(f"Merged image count: {written}")


if __name__ == "__main__":
    main()