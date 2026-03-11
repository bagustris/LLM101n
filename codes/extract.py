import re
import os
import pathlib

chapters_dir = pathlib.Path(".")
codes_dir = pathlib.Path("codes")

for n in range(1, 18):
    md_path = chapters_dir / f"ch{n:02d}.md"
    out_dir = codes_dir / f"ch{n:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_text = md_path.read_text()

    # Extract all ```python ... ``` blocks
    blocks = re.findall(r'```python\n(.*?)```', md_text, re.DOTALL)

    combined = "\n\n".join(blocks)

    # Fix data directory paths to be relative to codes/chNN/
    # Use regex to tolerate any amount of whitespace around the '=' sign
    combined = re.sub(r'DATA_DIR\s*=\s*"data"', 'DATA_DIR = "../data"', combined)
    combined = re.sub(r"DATA_DIR\s*=\s*'data'", "DATA_DIR = '../data'", combined)
    combined = combined.replace('"data/', '"../data/')
    combined = combined.replace("'data/", "'../data/")

    out_path = out_dir / "main.py"
    out_path.write_text(combined)
    print(f"Wrote {out_path} ({len(blocks)} blocks, {len(combined)} chars)")
