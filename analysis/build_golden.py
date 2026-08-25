# analysis/build_golden.py
"""Build the local golden fixture from the aggregated corpus.

Output is gitignored: it contains real prompt text. Run agg_sessions.py first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from router.classify import classify  # noqa: E402
from router.config import load_classes, load_settings  # noqa: E402

SRC = REPO / "analysis/sessions.json"
DST = REPO / "tests/fixtures/golden.local.jsonl"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing - run: uv run python analysis/agg_sessions.py 14")
    classes = load_classes(REPO / "config/classes.yaml", REPO / "config/classes.local.yaml")
    settings = load_settings(REPO / "config/settings.yaml")
    rows = json.loads(SRC.read_text())
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w") as fh:
        for row in rows:
            prompt = row["first_prompt"]
            guess = classify(prompt, classes, settings)
            fh.write(json.dumps({"prompt": prompt, "class": guess.cls}) + "\n")
    print(f"wrote {len(rows)} rows to {DST}")
    print("REVIEW EVERY LINE BY HAND: these are the classifier's own guesses, "
          "not labels. Correct the 'class' values before trusting the test.")


if __name__ == "__main__":
    main()
