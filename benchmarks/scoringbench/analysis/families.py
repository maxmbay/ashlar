"""A family for each of the 63 published models, with the evidence for each call.

Families:
  pretrained    the model's weights were trained on other data before the benchmark (a checkpoint is loaded), and
                they are not updated on the table: in-context learners, and conformal calibration around one;
  fine-tuned    a pretrained checkpoint whose weights are updated on the table;
  tuned         trained from scratch on the table, with a hyperparameter search on the table;
  untuned       trained from scratch on the table with fixed settings.

The classification is ours: results/model_families.json gives each model's family, a note, and the evidence, as
strings in ScoringBench's model registry and wrappers. With ScoringBench's code at 3bf48c9 (SCORINGBENCH_CODE), each
string is found in its file and turned into a line number, and organizations and links come from ScoringBench's
models.json; without it, those columns are left empty. ScoringBench itself labels no model by family.

Writes out/families.csv, out/tab_families.tex and the section ``families`` of out/numbers.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import pandas as pd  # noqa: E402

FAMILIES = json.loads((c.RESULTS / "model_families.json").read_text())
CODE = c.code_folder() is not None


def where(rel: str, needle: str) -> str:
    """'file:line' of the first line of ScoringBench's file that contains needle; stops if there is none."""
    for i, line in enumerate(c.code_file(rel).splitlines(), 1):
        if needle in line:
            return f"{rel}:{i}"
    raise SystemExit(f"evidence not found: {needle!r} in {rel}")


meta = {m["model_name"]: m for m in (c.models_json() or [])}
published = sorted(c.published())
assert set(published) == set(FAMILIES), set(published) ^ set(FAMILIES)
assert not meta or set(meta) == set(published)

rows = []
for model in published:
    entry = FAMILIES[model]
    links = (meta.get(model) or {}).get("links") or {}
    rows.append({
        "model": model, "family": entry["family"], "in_untuned_list": model in c.UNTUNED,
        "organization": (meta.get(model) or {}).get("organization") or "",
        "huggingface": links.get("huggingface") or "",
        "evidence": "; ".join(where(rel, needle) for rel, needle in entry["evidence"]) if CODE else "",
        "note": entry["note"],
    })
df = pd.DataFrame(rows)
order = {"pretrained": 0, "fine-tuned": 1, "tuned": 2, "untuned": 3}
df = df.sort_values(["family", "model"], key=lambda s: s.map(order) if s.name == "family" else s.str.lower())
c.write_csv("families.csv", df)

untuned_found = set(df[df.family == "untuned"].model)
listed = set(c.UNTUNED)
counts = df.family.value_counts().to_dict()
print(counts)
print("untuned by evidence == the list of 20:", untuned_found == listed,
      "| listed but not untuned:", sorted(listed - untuned_found), "| untuned but not listed:", sorted(untuned_found - listed))
if CODE:
    print("evidence strings found in ScoringBench's code at 3bf48c9 for all", len(df), "models")
    print("untuned models with a Hugging Face link:", sorted(df[(df.family == "untuned") & (df.huggingface != "")].model) or "none")
else:
    print("no clone of ScoringBench's code (analyze.py --code): evidence lines and organizations left empty")

labels = {"pretrained": "Pretrained, used in context or with a fitted calibration", "fine-tuned": "Fine-tuned from a pretrained model",
          "tuned": "Trained from scratch, tuned on each table", "untuned": "Trained from scratch, untuned"}
tex_rows, groups = [], {}
for fam in order:
    sub = df[df.family == fam]
    groups[len(tex_rows)] = f"{labels[fam]} ({len(sub)})"
    for r in sub.itertuples():
        tex_rows.append([c.tex_name(r.model), c.tex_escape(r.organization) or "--"])
c.write_tex("tab_families.tex", "ll", ["ScoringBench name", "Organization (models.json)"], tex_rows, long=True, groups=groups,
            notes="The 63 published ScoringBench models by family; evidence in out/families.csv. Script: analysis/families.py")
c.record("families", {"counts": counts, "untuned_list_matches_evidence": untuned_found == listed,
                      "listed_not_untuned": sorted(listed - untuned_found),
                      "untuned_not_listed": sorted(untuned_found - listed)})
