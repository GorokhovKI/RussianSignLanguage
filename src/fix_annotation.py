# scripts/fix_annotations_csv.py
import csv
import re
from pathlib import Path

import pandas as pd

SRC = Path('data/annotations.csv')
OUT = Path('data/annotations_fixed.csv')

uuid_re = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')

text = SRC.read_text(encoding='utf-8', errors='replace')

# insert newline before any UUID that is NOT at start of line
# pattern: any position where preceding char is not newline, but ahead is uuid
fixed = re.sub(r'(?<!\n)(' + uuid_re.pattern + r')', r'\n\1', text)

# remove possible leading empty line
if fixed.startswith('\n'):
    fixed = fixed[1:]

OUT.write_text(fixed, encoding='utf-8')
print(f'Wrote fixed file: {OUT} (original backed up as {SRC}.bak)')

try:
    df = pd.read_csv(OUT, sep=',', header=None, engine='python', on_bad_lines='error')
except pd.errors.ParserError as e:
    print("❌ Ошибка парсинга CSV:")
    print(e)
    # найдем номер строки из текста ошибки
    import re
    m = re.search(r'line (\d+)', str(e))
    if m:
        line_number = int(m.group(1))
        print(f"\n🔍 Проблемная строка {line_number}:")
        with open(OUT, encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                if abs(i - line_number) <= 2:  # покажем строку и соседние
                    print(f"{i}: {line.rstrip()}")
