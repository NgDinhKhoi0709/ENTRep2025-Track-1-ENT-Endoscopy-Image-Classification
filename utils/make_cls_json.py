import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def read_data_json(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def build_cls_mapping(records: List[Dict[str, Any]], *, skip_empty: bool = True) -> Dict[str, str]:
    cls_map: Dict[str, str] = {}
    for idx, item in enumerate(records):
        image_path = item.get("Path")
        classification = item.get("Classification")

        if image_path is None:
            continue
        if skip_empty and (classification is None or str(classification).strip() == ""):
            continue

        cls_map[str(image_path)] = str(classification) if classification is not None else ""
    return cls_map


def write_json_mapping(mapping: Dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cls.json from data.json (Path -> Classification)")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("dataset/train/data.json"),
        help="Path to data.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dataset/train/cls.json"),
        help="Path to write cls.json",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include entries with empty/missing Classification",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        records = read_data_json(args.input)
    except Exception as e:
        print(f"Lỗi đọc file input: {e}", file=sys.stderr)
        return 1

    mapping = build_cls_mapping(records, skip_empty=not args.include_empty)
    try:
        write_json_mapping(mapping, args.output)
    except Exception as e:
        print(f"Lỗi ghi file output: {e}", file=sys.stderr)
        return 2

    print(f"Đã tạo {args.output} với {len(mapping)} mục từ {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


