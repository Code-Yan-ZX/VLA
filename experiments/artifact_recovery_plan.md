# Anonymous artifact recovery plan

Status: blocked on access to the experiment server as of 2026-08-18.

## Audit result

- Supplement S9 names 53 unique run JSON files.
- The local clone contains none of those run JSON files and no `runs/` tree.
- Fourteen S9 audit-record classes still need concrete manifest entries.
- Six S10 qualitative panels (`img70,71,73,75,76,77`) are also absent.
- Existing evaluation/configuration files contain absolute server paths and must
  be sanitized before anonymous release.

## Recovery source

Expected server repository:

```text
/media/disk2/YZX/research/vla
```

SSH probes on 2026-08-18:

- `jnh3`: connection timeout.
- `192.168.3.69:60000`: connection timeout.
- `lightinfra-3`: reachable but requires credentials.

## Recovery command

After copying the server-side repository or a selective artifact export to a
local directory, run:

```powershell
.\scripts\build_anonymous_artifact.ps1 -SourceRoot D:\path\to\server-export
```

The script parses the S9 filenames from `supp.tex`, requires a unique match for
every JSON, stages selected code/configuration files, rejects known identity and
absolute-path strings, and writes `manifests/SHA256SUMS`.

## Submission gate

Do not describe the anonymous artifact as available until all of the following
are true:

- 53/53 S9 JSON files resolve uniquely.
- The 14 audit-record classes have machine-readable manifest entries.
- Per-sample predictions, official rescoring, paired bootstrap/permutation
  outputs, skip accounting, token counts, D1 records, and model/engine versions
  are present.
- The six missing S10 panels are restored.
- Dataset paths and ground truth comply with redistribution licenses.
- The anonymity scan is empty and checksums verify after packaging.
