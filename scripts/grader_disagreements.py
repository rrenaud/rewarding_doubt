"""List answers that F1 > 0.5 and exact-match grading disagree on, from a released-evaluation output.

    python scripts/grader_disagreements.py [runs/modal-stage2-20261001T082216Z/eval_base.json]

Both graders are verbatim ports of the released util/EvaluationMetrics.py (rewarding_doubt.paper_ppo).
"""
import json
import sys

from rewarding_doubt.paper_ppo import f1_score, is_correct_exact, is_correct_f1


def main(path):
    rows = [r for r in json.load(open(path)) if not r["is_wrong_format"]]
    diff = [r for r in rows if is_correct_f1(r["prediction"], r["gt_candidates"]) != is_correct_exact(r["prediction"], r["gt_candidates"])]
    print(f"{len(diff)} of {len(rows)} well-formed answers graded differently "
          f"({sum(is_correct_f1(r['prediction'], r['gt_candidates']) for r in diff)} F1-correct only)")
    for i, r in enumerate(diff):
        alias = max(r["gt_candidates"], key=lambda a: f1_score(r["prediction"], a))
        print(f"{i:2d} | {r['question'].strip()[:90]} | {r['prediction']} | {alias} | F1 {f1_score(r['prediction'], alias):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/modal-stage2-20261001T082216Z/eval_base.json")
