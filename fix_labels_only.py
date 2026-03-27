from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Stats:
    total_files_processed: int = 0
    total_files_fixed: int = 0
    total_skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix YOLO class IDs in-place for Burnt Socket and Exposed_Wire datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root dataset directory containing 'Burnt Socket' and 'Exposed_Wire'.",
    )
    parser.add_argument(
        "--warn",
        action="store_true",
        help="Print detailed warnings for skipped files and invalid lines.",
    )
    return parser.parse_args()


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_label_for_image(image_path: Path) -> Path | None:
    # 1) Same directory and stem: image.jpg -> image.txt
    same_dir_candidate = image_path.with_suffix(".txt")
    if same_dir_candidate.exists() and same_dir_candidate.is_file():
        return same_dir_candidate

    # 2) YOLO style sibling folders: .../images/.../a.jpg -> .../labels/.../a.txt
    parts_lower = [part.lower() for part in image_path.parts]
    if "images" in parts_lower:
        idx = parts_lower.index("images")
        parts = list(image_path.parts)
        parts[idx] = "labels"
        yolo_candidate = Path(*parts).with_suffix(".txt")
        if yolo_candidate.exists() and yolo_candidate.is_file():
            return yolo_candidate

    return None


def parse_and_remap_lines(lines: list[str], class_id: int, warn: bool, label_path: Path) -> list[str]:
    fixed_lines: list[str] = []

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        tokens = stripped.split()
        if len(tokens) != 5:
            if warn:
                print(f"[WARN] Invalid format in {label_path} line {line_number}: expected 5 tokens")
            continue

        try:
            # Validate YOLO numeric layout and keep bbox values unchanged.
            float(tokens[0])
            float(tokens[1])
            float(tokens[2])
            float(tokens[3])
            float(tokens[4])
        except ValueError:
            if warn:
                print(f"[WARN] Non-numeric values in {label_path} line {line_number}")
            continue

        fixed_lines.append(f"{class_id} {tokens[1]} {tokens[2]} {tokens[3]} {tokens[4]}")

    return fixed_lines


def fix_label_file(label_path: Path, class_id: int, warn: bool) -> bool:
    try:
        raw_lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        if warn:
            print(f"[WARN] Failed to read {label_path}: {exc}")
        return False

    if not any(line.strip() for line in raw_lines):
        if warn:
            print(f"[WARN] Empty label file: {label_path}")
        return False

    fixed_lines = parse_and_remap_lines(raw_lines, class_id, warn, label_path)
    if not fixed_lines:
        if warn:
            print(f"[WARN] No valid YOLO rows in: {label_path}")
        return False

    try:
        label_path.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")
    except OSError as exc:
        if warn:
            print(f"[WARN] Failed to write {label_path}: {exc}")
        return False

    return True


def process_class_folder(class_root: Path, class_id: int, warn: bool, stats: Stats) -> None:
    seen_labels: set[Path] = set()

    for image_path in iter_images(class_root):
        label_path = find_label_for_image(image_path)
        if label_path is None:
            stats.total_skipped += 1
            if warn:
                print(f"[WARN] Missing label for image: {image_path}")
            continue

        resolved_label = label_path.resolve()
        if resolved_label in seen_labels:
            continue
        seen_labels.add(resolved_label)

        stats.total_files_processed += 1
        if fix_label_file(label_path, class_id, warn):
            stats.total_files_fixed += 1
        else:
            stats.total_skipped += 1


def validate_structure(dataset_root: Path) -> tuple[Path, Path]:
    burnt_socket = dataset_root / "Burnt Socket"
    exposed_wire = dataset_root / "Exposed_Wire"

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    if not burnt_socket.exists() or not burnt_socket.is_dir():
        raise SystemExit(f"Missing folder: {burnt_socket}")
    if not exposed_wire.exists() or not exposed_wire.is_dir():
        raise SystemExit(f"Missing folder: {exposed_wire}")

    return burnt_socket, exposed_wire


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    burnt_socket_root, exposed_wire_root = validate_structure(dataset_root)
    stats = Stats()

    # Burnt Socket -> class 0 (burn_marks)
    process_class_folder(burnt_socket_root, class_id=0, warn=args.warn, stats=stats)
    # Exposed_Wire -> class 1 (exposed_wire)
    process_class_folder(exposed_wire_root, class_id=1, warn=args.warn, stats=stats)

    print(f"Total files processed: {stats.total_files_processed}")
    print(f"Total files fixed: {stats.total_files_fixed}")
    print(f"Total skipped: {stats.total_skipped}")


if __name__ == "__main__":
    main()