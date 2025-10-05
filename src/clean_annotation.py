#!/usr/bin/env python3
import csv
from pathlib import Path

src = Path('data/annotations.csv')
out_good = Path('data/annotations_clean.csv')
out_bad = Path('data/annotations_bad_lines.csv')

# пробуем определить разделитель (по sample)
with open(src, 'r', encoding='utf-8', errors='ignore') as f:
    sample = f.read(8192)
import sys
try:
    dialect = csv.Sniffer().sniff(sample)
    delim = dialect.delimiter
except Exception:
    delim = ','

good = []
bad = []
with open(src, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f, delimiter=delim)
    header = next(reader, None)
    # если header содержит более 1 поля, считаем его header; иначе — keep raw
    header_len = len(header) if header is not None else None
    if header_len is None:
        print('Empty file?')
        sys.exit(1)
    good.append(header)
    for i, row in enumerate(reader, start=2):
        if len(row) == header_len:
            good.append(row)
        else:
            bad.append((i, row))

# записываем
with open(out_good, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=delim)
    writer.writerows(good)
with open(out_bad, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # сохраняем номер строки и её содержимое
    writer.writerow(['line_number','raw'])
    for ln, row in bad:
        writer.writerow([ln, delim.join(row)])
print(f'Wrote {len(good)} good rows to {out_good}; {len(bad)} bad rows to {out_bad}')
