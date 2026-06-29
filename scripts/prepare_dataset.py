import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

_SPLIT_FILE_RE = re.compile(r'(.+)_test_split([123])\.txt$')
_SPLIT_LABELS = {0: 'unused', 1: 'train', 2: 'test'}


def _read_classes(data_root):
    classes = sorted(p.name for p in Path(data_root).iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(f'Nessuna cartella classe trovata in {data_root}')
    return classes


def _parse_split_file(split_file, data_root, class_to_index):
    split_file = Path(split_file)
    match = _SPLIT_FILE_RE.match(split_file.name)
    if not match:
        return []
    class_name, split_id_str = match.groups()
    records = []
    with split_file.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            file_name, split_val_str = line.split()[:2]
            split_val = int(split_val_str)
            video_path = (Path(data_root) / class_name / file_name).resolve()
            records.append({
                'path': str(video_path),
                'label': class_to_index[class_name],
                'class_name': class_name,
                'subset': _SPLIT_LABELS[split_val],
                'split_id': int(split_id_str),
                'file_name': file_name,
                'exists': video_path.exists(),
            })
    return records


def _source_group(record):
    """Nome del video sorgente di un clip HMDB51.

    I clip estratti dallo stesso filmato condividono il prefisso del filename
    fino al tag della classe (es. '1989_Tour_de_France_..._ride_bike_..._2.avi'
    -> '1989_Tour_de_France_...'). Raggruppare per sorgente evita il data leakage
    nello split di validation: clip quasi identici non devono finire metà in
    train e metà in val.
    """
    file_name = record['file_name']
    marker = f"_{record['class_name']}_"
    idx = file_name.find(marker)
    return file_name[:idx] if idx != -1 else file_name


def _stratified_val_split(train_records, val_ratio, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in train_records:
        by_class[r['class_name']].append(r)
    final_train, final_val = [], []
    for cls in sorted(by_class):
        # Raggruppa i clip per video sorgente, poi mescola i GRUPPI (non i clip)
        # cosi' tutti i clip di uno stesso filmato restano nello stesso subset.
        groups = defaultdict(list)
        for r in by_class[cls]:
            groups[_source_group(r)].append(r)
        keys = list(groups)
        rng.shuffle(keys)
        n_val = int(round(len(keys) * val_ratio))
        if len(keys) > 1:
            n_val = max(1, min(n_val, len(keys) - 1))
        else:
            n_val = 0
        val_keys = set(keys[:n_val])
        for key in keys:
            subset = 'val' if key in val_keys else 'train'
            target = final_val if key in val_keys else final_train
            for r in groups[key]:
                target.append({**r, 'subset': subset})
    rng.shuffle(final_train)
    rng.shuffle(final_val)
    return final_train, final_val


def _write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['path', 'label', 'class_name', 'subset', 'split_id', 'file_name']
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({f: r[f] for f in fields})


def build_manifests(data_root, splits_root, manifest_dir, split_id=1, val_ratio=0.15, seed=42):
    data_root = Path(data_root).resolve()
    splits_root = Path(splits_root).resolve()
    manifest_dir = Path(manifest_dir).resolve()

    classes = _read_classes(data_root)
    class_to_index = {c: i for i, c in enumerate(classes)}

    all_records = []
    for sf in sorted(splits_root.glob(f'*_test_split{split_id}.txt')):
        all_records.extend(_parse_split_file(sf, data_root, class_to_index))
    if not all_records:
        raise FileNotFoundError(f'Nessuno split file trovato in {splits_root}')

    missing = [r for r in all_records if not r['exists']]
    existing = [r for r in all_records if r['exists']]
    official_train = [r for r in existing if r['subset'] == 'train']
    official_test = [r for r in existing if r['subset'] == 'test']
    unused = [r for r in existing if r['subset'] == 'unused']

    train, val = _stratified_val_split(official_train, val_ratio, seed)
    test = [dict(r) for r in official_test]

    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / 'classes.txt').open('w', encoding='utf-8') as fh:
        fh.write('\n'.join(classes) + '\n')

    _write_csv(manifest_dir / f'split{split_id}_train.csv', train)
    _write_csv(manifest_dir / f'split{split_id}_val.csv', val)
    _write_csv(manifest_dir / f'split{split_id}_test.csv', test)
    _write_csv(manifest_dir / f'split{split_id}_official_train.csv', official_train)
    _write_csv(manifest_dir / f'split{split_id}_unused.csv', unused)

    counts = {
        'official': dict(Counter(r['subset'] for r in existing)),
        'generated': {'train': len(train), 'val': len(val), 'test': len(test), 'unused': len(unused)},
        'classes': len(classes),
        'missing_files': len(missing),
        'total_existing_in_split_files': len(existing),
    }
    summary = {'counts': counts}
    with (manifest_dir / f'split{split_id}_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    return summary
