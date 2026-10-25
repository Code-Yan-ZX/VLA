#!/usr/bin/env python
"""Extract a pre-fixed ID subset from a full prediction JSON into a sibling
prediction file (same schema) so curve points share exactly the same IDs and
denominators (directive 11/30). Usage:
  cvm24h_extract_slice.py full.json ids.txt out.json
"""
import json, sys
full, idf, out = sys.argv[1:4]
ids = {x.strip() for x in open(idf) if x.strip()}
d = json.load(open(full))
rows = [p for p in d["per_sample"] if str(p["id"]) in ids]
assert len(rows) == len(ids), f"{len(rows)} found vs {len(ids)} wanted"
missing = ids - {str(p["id"]) for p in rows}
assert not missing, sorted(missing)[:10]
d["per_sample"] = rows
d["n"] = len(rows)
d["slice_of"] = full
d["slice_ids_sha16"] = __import__("hashlib").sha256(
    "\n".join(sorted(ids)).encode()).hexdigest()[:16]
d.pop("acc", None); d.pop("mean_ptid_len", None)
json.dump(d, open(out, "w"), indent=1)
print(out, "n=", len(rows))
