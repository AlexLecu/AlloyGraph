"""Clean corrupted elasticity data from evaluation datasets.

Problem: Some alloys have shear modulus (G ≈ 70-85 GPa at RT) mixed in with
or replacing Young's modulus (E ≈ 190-230 GPa at RT) in the 'elasticity' field.

Rules:
1. RT value (≤30°C) < 120 GPa with no high values → entire series is shear modulus → clear
2. Mixed high/low values → remove entries < 100 GPa (definitely G, not E)
3. Smooth degradation curves (like 200→117 at 1000°C) are preserved
"""
import json
import sys
import os
from pathlib import Path


def clean_elasticity(entries: list) -> tuple[list, str]:
    """Clean elasticity entries. Returns (cleaned_entries, action_taken)."""
    if not entries:
        return entries, "no_data"

    vals = [(float(e['temp_c']), e['value']) for e in entries]
    rt_vals = [v for t, v in vals if t <= 30]
    high_vals = [v for _, v in vals if v >= 150]
    low_vals = [v for _, v in vals if v < 100]

    # Case 1: All values are low — entire series is shear modulus
    if not high_vals and all(v < 120 for _, v in vals):
        return [], "cleared_all_shear_modulus"

    # Case 2: Mixed — has both high (E) and low (G) values
    if high_vals and low_vals:
        cleaned = [e for e in entries if e['value'] >= 100]
        removed = len(entries) - len(cleaned)
        return cleaned, f"removed_{removed}_shear_modulus_entries"

    # Case 3: RT has both high and low (different datasheets mixed)
    if rt_vals and any(v < 120 for v in rt_vals) and any(v >= 150 for v in rt_vals):
        cleaned = [e for e in entries if e['value'] >= 100]
        removed = len(entries) - len(cleaned)
        return cleaned, f"removed_{removed}_mixed_rt_entries"

    return entries, "ok"


def process_file(filepath: str, dry_run: bool = False) -> dict:
    """Process a single JSONL file. Returns stats."""
    records = []
    changes = []

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    for rec in records:
        em = rec.get('elasticity', [])
        if not em:
            continue

        cleaned, action = clean_elasticity(em)
        if action not in ("ok", "no_data"):
            alloy = rec.get('alloy', '?')
            original_count = len(em)
            new_count = len(cleaned)
            changes.append(f"  {alloy}: {action} ({original_count} → {new_count} entries)")
            if not dry_run:
                rec['elasticity'] = cleaned

    if not dry_run and changes:
        with open(filepath, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return {"file": os.path.basename(filepath), "changes": changes}


def main():
    dry_run = "--dry-run" in sys.argv

    data_dir = Path(__file__).parent.parent / "data_clean"
    files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        print(f"No JSONL files found in {data_dir}")
        sys.exit(1)

    print(f"{'DRY RUN — ' if dry_run else ''}Cleaning elasticity data in {data_dir}")
    print("=" * 60)

    total_changes = 0
    for fpath in files:
        result = process_file(str(fpath), dry_run=dry_run)
        if result["changes"]:
            print(f"\n{result['file']}:")
            for c in result["changes"]:
                print(c)
            total_changes += len(result["changes"])
        else:
            print(f"{result['file']}: clean")

    print(f"\n{'Would fix' if dry_run else 'Fixed'} {total_changes} alloys")
    if dry_run:
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
