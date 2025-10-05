# scripts/clean_annotations.py
import csv
from pathlib import Path

SRC = Path('data/annotations.csv')
OUT = Path('data/annotations_cleaned.csv')
BAK = Path('data/annotations.csv.bak')

if not SRC.exists():
    raise SystemExit(f"Source not found: {SRC}")

# backup
if not BAK.exists():
    SRC.replace(BAK)   # rename original to .bak, we'll recreate SRC as cleaned
    SRC = BAK
else:
    # if .bak already exists, keep using SRC (do not overwrite backup)
    pass

bad_lines = []
total = 0
accepted = 0

with SRC.open('r', encoding='utf-8', errors='replace') as fin, OUT.open('w', encoding='utf-8', newline='') as fout:
    writer = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i, raw in enumerate(fin, 1):
        line = raw.rstrip('\n\r')
        if not line.strip():
            continue
        total += 1
        # try tab split first
        parts_tab = line.split('\t')
        if len(parts_tab) >= 7:
            parts = parts_tab[:7]
            writer.writerow(parts)
            accepted += 1
            continue
        # else try comma split with max 6 splits => at most 7 parts
        parts_comma = line.split(',', 6)
        if len(parts_comma) == 7:
            writer.writerow(parts_comma)
            accepted += 1
            continue
        # fallback: try csv.reader (handles quoted commas)
        try:
            parsed = next(csv.reader([line], delimiter=',', quotechar='"'))
            if len(parsed) == 7:
                writer.writerow(parsed)
                accepted += 1
                continue
        except Exception:
            pass
        # if still not ok — record bad
        bad_lines.append((i, line))
        # attempt a last-resort: split by any whitespace (tab/space) and take first 7 (risky)
        parts_ws = line.split()
        if len(parts_ws) >= 7:
            writer.writerow(parts_ws[:7])
            accepted += 1
            continue

# report
print(f"Total lines scanned: {total}")
print(f"Accepted lines: {accepted}")
print(f"Bad lines: {len(bad_lines)} (first 20 shown)")
for ln, content in bad_lines[:20]:
    print(f"{ln}: {content}")
if len(bad_lines) > 0:
    print("Bad lines were not written to cleaned file; fix them manually in the backup file and re-run.")
else:
    print(f"Cleaned file written to: {OUT}")
    print("You can now replace original file if you want:")
    print(f"  mv {OUT} data/annotations.csv")
