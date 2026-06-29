from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_FILE_RE = re.compile(r"(.+)_test_split([123])\.txt$")
SPLIT_LABELS = {0: "unused", 1: "train", 2: "test"}


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def has_avi_files(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.avi"))


def find_archives(root: Path, patterns: list[str]) -> list[Path]:
    archives: list[Path] = []
    for pattern in patterns:
        archives.extend(root.glob(pattern))
    return sorted(set(path for path in archives if path.is_file()))


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(destination)
        return

    bsdtar = shutil.which("bsdtar")
    unrar = shutil.which("unrar")
    seven_zip = shutil.which("7z")
    unar = shutil.which("unar")

    if bsdtar:
        run([bsdtar, "-xf", str(archive), "-C", str(destination)])
    elif unrar:
        run([unrar, "x", "-o+", str(archive), str(destination)])
    elif seven_zip:
        run([seven_zip, "x", "-y", f"-o{destination}", str(archive)])
    elif unar:
        run([unar, "-force-overwrite", "-output-directory", str(destination), str(archive)])
    else:
        raise RuntimeError(
            "Non trovo un estrattore per archivi .rar. Installa uno tra: bsdtar, unrar, 7z oppure unar."
        )


def find_video_root(root: Path) -> Path:
    candidates = [
        root / "dataset" / "hmdb51",
        root / "dataset" / "hmdb51_org",
        root / "hmdb51",
        root / "hmdb51_org",
        root / "HMDB51",
    ]
    for candidate in candidates:
        if candidate.exists() and (has_avi_files(candidate) or list(candidate.glob("*.rar"))):
            return candidate

    archives = find_archives(
        root,
        [
            "hmdb51_org.rar",
            "hmdb51_org.zip",
            "*HMDB*.rar",
            "*HMDB*.zip",
            "*hmdb*.rar",
            "*hmdb*.zip",
        ],
    )
    archives = [path for path in archives if "split" not in path.name.lower()]
    if not archives:
        raise FileNotFoundError(
            "Non trovo il dataset HMDB51. Rimetti nella cartella questo file/cartella: "
            "hmdb51.zip oppure hmdb51/."
        )

    print(f"Estraggo dataset video da: {archives[0].name}")
    extract_archive(archives[0], root)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Ho estratto l'archivio, ma non trovo la cartella hmdb51.")


def extract_nested_class_archives(video_root: Path) -> None:
    class_archives = sorted(video_root.glob("*.rar")) + sorted(video_root.glob("*.zip"))
    if not class_archives:
        return

    for archive in class_archives:
        class_dir = video_root / archive.stem
        if has_avi_files(class_dir):
            print(f"Salto {archive.name}: classe gia' estratta.")
            continue
        print(f"Estraggo classe: {archive.name}")
        extract_archive(archive, video_root)


def find_splits_root(root: Path) -> Path:
    candidates = [
        root / "dataset" / "testTrainMulti_7030_splits",
        root / "testTrainMulti_7030_splits",
        root / "test_train_splits",
        root / "splits",
    ]
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("*_test_split1.txt")):
            return candidate

    if list(root.glob("*_test_split1.txt")):
        return root

    archives = find_archives(
        root,
        [
            "test_train_splits.rar",
            "test_train_splits.zip",
            "testTrainMulti_7030_splits.rar",
            "testTrainMulti_7030_splits.zip",
            "*split*.rar",
            "*split*.zip",
        ],
    )
    if not archives:
        raise FileNotFoundError(
            "Non trovo gli split ufficiali. Rimetti nella cartella questo file/cartella: "
            "test_train_splits.rar oppure testTrainMulti_7030_splits/."
        )

    print(f"Estraggo split da: {archives[0].name}")
    extract_archive(archives[0], root)

    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("*_test_split1.txt")):
            return candidate
    if list(root.glob("*_test_split1.txt")):
        return root

    raise FileNotFoundError("Ho estratto gli split, ma non trovo i file *_test_split1.txt.")


def try_find_splits_root(root: Path) -> Path | None:
    try:
        return find_splits_root(root)
    except FileNotFoundError as exc:
        print(f"Split ufficiali non trovati: {exc}")
        print("Creo split stratificati automatici dal dataset video.")
        return None


def read_classes(data_root: Path) -> list[str]:
    classes = sorted(path.name for path in data_root.iterdir() if path.is_dir() and has_avi_files(path))
    if not classes:
        raise FileNotFoundError(f"Nessuna classe video trovata in {data_root}")
    return classes


def parse_split_file(split_file: Path, data_root: Path, class_to_index: dict[str, int]) -> list[dict[str, object]]:
    match = SPLIT_FILE_RE.match(split_file.name)
    if not match:
        return []

    class_name, split_id_text = match.groups()
    split_id = int(split_id_text)
    if class_name not in class_to_index:
        print(f"Attenzione: split {split_file.name} ignorato, classe non trovata nei video.")
        return []

    records: list[dict[str, object]] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            file_name = parts[0]
            split_value = int(parts[1])
            video_path = (data_root / class_name / file_name).resolve()
            records.append(
                {
                    "path": str(video_path),
                    "label": class_to_index[class_name],
                    "class_name": class_name,
                    "subset": SPLIT_LABELS[split_value],
                    "split_id": split_id,
                    "file_name": file_name,
                    "exists": video_path.exists(),
                }
            )
    return records


def stratified_validation_split(
    train_records: list[dict[str, object]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in train_records:
        by_class[str(record["class_name"])].append(record)

    final_train: list[dict[str, object]] = []
    final_val: list[dict[str, object]] = []

    for class_name in sorted(by_class):
        records = list(by_class[class_name])
        rng.shuffle(records)
        val_count = int(round(len(records) * val_ratio))
        if len(records) > 1:
            val_count = max(1, min(val_count, len(records) - 1))
        else:
            val_count = 0

        for record in records[:val_count]:
            item = dict(record)
            item["subset"] = "val"
            final_val.append(item)
        for record in records[val_count:]:
            item = dict(record)
            item["subset"] = "train"
            final_train.append(item)

    rng.shuffle(final_train)
    rng.shuffle(final_val)
    return final_train, final_val


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "class_name", "subset", "split_id", "file_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def build_random_manifests(
    data_root: Path,
    manifest_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    classes = read_classes(data_root)
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    train: list[dict[str, object]] = []
    val: list[dict[str, object]] = []
    test: list[dict[str, object]] = []

    for class_name in classes:
        videos = sorted((data_root / class_name).glob("*.avi"))
        rng.shuffle(videos)
        total = len(videos)
        test_count = max(1, int(round(total * test_ratio))) if total > 2 else 0
        val_count = max(1, int(round(total * val_ratio))) if total - test_count > 1 else 0
        if total - test_count - val_count <= 0 and total > 0:
            val_count = max(0, val_count - 1)

        subsets = {
            "test": videos[:test_count],
            "val": videos[test_count : test_count + val_count],
            "train": videos[test_count + val_count :],
        }
        for subset, paths in subsets.items():
            for video_path in paths:
                record = {
                    "path": str(video_path.resolve()),
                    "label": class_to_index[class_name],
                    "class_name": class_name,
                    "subset": subset,
                    "split_id": 0,
                    "file_name": video_path.name,
                }
                if subset == "train":
                    train.append(record)
                elif subset == "val":
                    val.append(record)
                else:
                    test.append(record)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "classes.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(classes) + "\n")

    write_csv(manifest_dir / "split0_train.csv", train)
    write_csv(manifest_dir / "split0_val.csv", val)
    write_csv(manifest_dir / "split0_test.csv", test)

    # Duplica anche come split1 per compatibilita' con il notebook.
    write_csv(manifest_dir / "split1_train.csv", train)
    write_csv(manifest_dir / "split1_val.csv", val)
    write_csv(manifest_dir / "split1_test.csv", test)

    summary = {
        "data_root": str(data_root.resolve()),
        "splits_root": None,
        "manifest_dir": str(manifest_dir.resolve()),
        "split_strategy": "random_stratified",
        "generated": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "classes": len(classes),
        "total_videos": len(train) + len(val) + len(test),
    }
    with (manifest_dir / "split0_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (manifest_dir / "split1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_manifests(
    data_root: Path,
    splits_root: Path,
    manifest_dir: Path,
    split_id: int,
    val_ratio: float,
    seed: int,
) -> dict[str, object]:
    classes = read_classes(data_root)
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    all_records: list[dict[str, object]] = []
    for split_file in sorted(splits_root.glob(f"*_test_split{split_id}.txt")):
        all_records.extend(parse_split_file(split_file, data_root, class_to_index))

    if not all_records:
        raise FileNotFoundError(f"Nessuno split file valido trovato in {splits_root}")

    missing = [record for record in all_records if not bool(record["exists"])]
    existing = [record for record in all_records if bool(record["exists"])]
    official_train = [record for record in existing if record["subset"] == "train"]
    test = [dict(record) for record in existing if record["subset"] == "test"]
    unused = [dict(record) for record in existing if record["subset"] == "unused"]
    train, val = stratified_validation_split(official_train, val_ratio=val_ratio, seed=seed)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "classes.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(classes) + "\n")

    write_csv(manifest_dir / f"split{split_id}_train.csv", train)
    write_csv(manifest_dir / f"split{split_id}_val.csv", val)
    write_csv(manifest_dir / f"split{split_id}_test.csv", test)
    write_csv(manifest_dir / f"split{split_id}_official_train.csv", official_train)
    write_csv(manifest_dir / f"split{split_id}_unused.csv", unused)

    summary = {
        "data_root": str(data_root.resolve()),
        "splits_root": str(splits_root.resolve()),
        "manifest_dir": str(manifest_dir.resolve()),
        "official": dict(Counter(str(record["subset"]) for record in existing)),
        "generated": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "unused": len(unused),
        },
        "classes": len(classes),
        "missing_files": len(missing),
        "total_existing_in_split_files": len(existing),
    }
    with (manifest_dir / f"split{split_id}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trova HMDB51 riscaricato nella cartella, lo estrae se serve e crea i manifest CSV."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Cartella dove hai rimesso dataset e split.")
    parser.add_argument("--split-id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_dir = args.manifest_dir or (root / "data" / "manifests")

    print(f"Cartella progetto: {root}")
    video_root = find_video_root(root)
    extract_nested_class_archives(video_root)
    splits_root = try_find_splits_root(root)

    if splits_root is None:
        summary = build_random_manifests(
            data_root=video_root,
            manifest_dir=manifest_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        summary = build_manifests(
            data_root=video_root,
            splits_root=splits_root,
            manifest_dir=manifest_dir,
            split_id=args.split_id,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    print("\nDataset pronto.")
    print(json.dumps(summary, indent=2))
    print("\nOra puoi aprire il notebook e lasciare questi percorsi:")
    print(f"DATA_ROOT = Path({str(video_root)!r})")
    if splits_root is None:
        print("SPLITS_ROOT = None")
    else:
        print(f"SPLITS_ROOT = Path({str(splits_root)!r})")
    print(f"MANIFEST_DIR = Path({str(manifest_dir)!r})")


if __name__ == "__main__":
    main()
