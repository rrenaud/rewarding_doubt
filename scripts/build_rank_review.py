"""The rank 4 vs rank 8 evidence review: one web page with charts, from runs/minimal/rank_review_data.json.

    python scripts/build_rank_review.py OUT.html

Data (gathered from the run logs): the offline late-half LoRA rank sweep (ranks 1, 2, 4 and 8, 500 steps, 2 seeds),
the two online F1-label runs that differ in rank (rank 8 and rank 4, one seed each, to 2,000 steps), and the released
evaluation of their adapters on the 512 dev questions every 250 steps.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "runs/minimal/rank_review_data.json").read_text())
    page = (ROOT / "scripts/rank_review_template.html").read_text().replace("__DATA__", json.dumps(data))
    Path(sys.argv[1]).write_text(page)
    print(f"-> {sys.argv[1]} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
