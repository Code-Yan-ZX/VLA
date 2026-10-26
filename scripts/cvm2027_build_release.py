#!/usr/bin/env python3
"""Build the completed anonymous CVM review draft and a minimal source archive.

Nothing is copied to canonical artifact paths until source checks, compilation,
PDF checks, clean-source recompilation, and page rendering all succeed. This
script never uploads or submits a paper. Run --check-source-only while results
are pending; that mode writes no build or release artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile


DEFAULT_ENGINE = Path("/tmp/cvm-tectonic-bin/tectonic")
PAPER_REL = Path("drafts/cvm2027_submission_20261025")
TEX_NAMES = {"main.tex", "figure1_review.tex", "figure4_review.tex"}
BASE_NAMES = {"main.tex", "references.bib", "README.md", "cvm.sty",
              "cvm.bst", "cvm_eso.sty", "eso-pic.sty"}
FIGURE_NAMES = {
    "figs/fig1_user_20260915_v2.png", "figs/fig2_word.png",
    "figs/fig4_vector.pdf", "figs/budget_curves.pdf",
    "figs/score_vs_latency.pdf",
}
UNFINISHED = re.compile(r"\b(?:pending|todo|tbd|fixme|await|awaiting)\b", re.IGNORECASE)
TEX_FAILURES = (
    r"Overfull\s+\\[hv]box",
    r"Missing character:",
    r"(?:Citation|Reference)\s+[^\n]*undefined",
    r"There were undefined (?:references|citations)",
    r"(?:LaTeX|Package [^\n]+) Error:",
    r"^! ",
)


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str], *, cwd: Path | None = None,
            env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(args, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(completed.returncode == 0,
            f"Command failed ({completed.returncode}): {args!r}\n{completed.stdout[-6000:]}")
    return completed.stdout


def uncomment(text: str) -> str:
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0]
                     for line in text.splitlines())


def local_source(paper: Path, relative: str) -> Path:
    path = paper / relative
    require(path.is_file() and not path.is_symlink(),
            f"Missing, non-regular, or symlinked source file: {relative}")
    require(path.resolve().is_relative_to(paper.resolve()),
            f"Source escapes the manuscript directory: {relative}")
    return path


def source_manifest(paper: Path, expected_citations: int) -> dict:
    """Follow only explicitly allowed TeX inputs and figure assets."""
    pending = ["main.tex"]
    tex: dict[str, str] = {}
    figures: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in tex:
            continue
        require(relative in TEX_NAMES, f"TeX input is outside the whitelist: {relative}")
        body = uncomment(local_source(paper, relative).read_text())
        match = UNFINISHED.search(body)
        require(not match, f"Unfinished manuscript marker in {relative}: {match.group(0) if match else ''}")
        tex[relative] = body
        for child in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", body):
            pending.append(child if child.endswith(".tex") else child + ".tex")
        for figure in re.findall(r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}", body):
            require(figure in FIGURE_NAMES, f"Figure is outside the whitelist: {figure}")
            figures.add(figure)

    combined = "\n".join(tex.values())
    main = tex["main.tex"]
    require(re.search(r"\\author\s*\{\s*\}", main) is not None,
            "Anonymous review source must retain an empty author field")
    require(re.search(r"pdfauthor\s*=\s*\{\s*\}", main) is not None,
            "Anonymous review source must set an empty PDF author")
    for pattern in (r"\\cvmfinalcopy\b", r"\\(?:affiliation|institute|email|thanks)\s*\{",
                    r"\\(?:section|subsection)\*?\s*\{Acknowledg"):
        require(re.search(pattern, combined, re.IGNORECASE) is None,
                f"Review source contains an identifying/final-copy construct: {pattern}")
    labels = re.findall(r"\\label\s*\{([^}]+)\}", combined)
    references = set(re.findall(r"\\(?:ref|pageref|autoref)\s*\{([^}]+)\}", combined))
    require(len(labels) == len(set(labels)), "Duplicate TeX labels")
    require(not references - set(labels), f"Missing TeX labels: {sorted(references - set(labels))}")
    require(re.search(r"\\nocite\b", combined) is None,
            "Bibliography must contain actual citations rather than nocite entries")
    citations = {key.strip() for group in re.findall(r"\\cite\s*(?:\[[^]]*\])?\s*\{([^}]+)\}", combined)
                 for key in group.split(",")}
    bibliography = local_source(paper, "references.bib").read_text()
    entries = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibliography)
    require(len(entries) == len(set(entries)), "Duplicate bibliography keys")
    require(citations == set(entries),
            f"Citation mismatch: missing={sorted(citations - set(entries))}; uncited={sorted(set(entries) - citations)}")
    require(len(citations) == expected_citations,
            f"Expected {expected_citations} actual citations; found {len(citations)}")
    require(4 <= len(figures) <= 5, f"Expected four or five used figures; found {len(figures)}")
    names = sorted(BASE_NAMES | set(tex) | figures)
    hashes = {name: sha256(local_source(paper, name)) for name in names}
    return {"files": names, "sha256": hashes, "citation_count": len(citations),
            "citation_keys": sorted(citations), "figure_count": len(figures),
            "figures": sorted(figures), "label_count": len(labels)}


def anonymity_terms(repo: Path) -> list[str]:
    """Known project identifiers only; cited authors are legitimate bibliography content."""
    terms = ["/media/disk2/YZX", "/home/dell", "Code-Yan-ZX/VLA"]
    result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if result.returncode == 0:
        remote = result.stdout.strip().removesuffix(".git")
        match = re.search(r"github\.com[:/]([^/]+/[^/]+)$", remote)
        if match:
            terms.append(match.group(1))
    return sorted(set(terms))


def release_evidence(repo: Path) -> dict:
    """Require the complete audited experiment reports, without rescoring data."""
    matched_path = repo / "reports/cvm2027_matched_main_20261026.json"
    cost_path = repo / "reports/cvm2027_corrected_quality_cost.json"
    for path in (matched_path, cost_path):
        require(path.is_file(), f"Required completed experiment report is absent: {path.name}")
    matched = json.loads(matched_path.read_text())
    cost = json.loads(cost_path.read_text())
    attention = {"vision_config": "sdpa", "text_config": "eager"}
    require(matched.get("complete") is True and matched.get("cell_count") == 8,
            "The shared-attention full-split report is incomplete")
    validation = matched.get("validation", {})
    require(validation.get("complete") is True and validation.get("protocol_equal_all_eight") is True,
            "Full-split protocol validation did not pass")
    require(matched["protocol"]["attention"] == attention,
            "Full-split results do not use the intended shared attention protocol")
    require(matched["statistics"]["resamples"] == 20000,
            "Full-split paired intervals do not use the required resample count")
    raw_sources = {}
    for benchmark, expected_n in (("textvqa", 5000), ("ocrbench", 1000)):
        block = matched["benchmarks"][benchmark]
        require(block["n_attempted"] == expected_n and len(block["pairs"]) == 6,
                f"Missing full-split attempts or planned pairs: {benchmark}")
        require(validation["input_and_budget"][benchmark]["mismatches"] == 0,
                f"Full-split input/budget mismatches: {benchmark}")
        require(set(block["cells"]) == {"none", "pre", "fastv", "hilo"},
                f"Missing full-split method: {benchmark}")
        for mode, cell in block["cells"].items():
            require(cell["n_attempted"] == expected_n and
                    cell["n_answered"] + cell["n_failed"] == expected_n,
                    f"Incomplete full-split denominator: {benchmark}/{mode}")
            source = matched["sources"][benchmark][mode]
            path = Path(source["path"])
            require(path.is_absolute(), f"Unresolved prediction provenance: {path}")
            require(path.is_file() and sha256(path) == source["sha256"],
                    f"Full-split prediction changed after scoring: {benchmark}/{mode}")
            raw_sources[f"{benchmark}/{mode}"] = source["sha256"]
    expected_tags = {"none"} | {f"{mode}_{keep}" for mode in ("pre", "fastv", "hilo")
                               for keep in (.25, .5, .75)}
    require(cost.get("validation", {}).get("complete") is True and
            set(cost.get("configs", {})) == expected_tags,
            "The corrected ten-configuration quality/cost report is incomplete")
    for tag, benchmarks in cost["configs"].items():
        require(set(benchmarks) == {"textvqa", "ocrbench"}, f"Missing cost task: {tag}")
        for benchmark, cell in benchmarks.items():
            require(cell["n"] == cell["n_success"] == 100,
                    f"Incomplete same-request quality/cost subset: {tag}/{benchmark}")
            require(cell["paired_vs_none"]["resamples"] == 20000,
                    f"Missing paired cost-subset interval: {tag}/{benchmark}")
    for engine, source in cost["provenance"].items():
        require(source["attention"] == attention, f"Corrected timing attention differs: {engine}")
        path = Path(source["path"])
        require(path.is_absolute(), f"Unresolved timing provenance: {path}")
        require(path.is_file() and sha256(path) == source["sha256"],
                f"Corrected timing changed after scoring: {engine}")
        raw_sources[f"timing/{engine}"] = source["sha256"]
    return {"matched_main": {"path": str(matched_path), "sha256": sha256(matched_path),
                             "complete": True, "cell_count": 8, "planned_pair_count": 12},
            "corrected_quality_cost": {"path": str(cost_path), "sha256": sha256(cost_path),
                                       "complete": True, "cell_count": 10,
                                       "quality_examples_per_task": 100},
            "verified_raw_sha256": raw_sources}


def page_text(pdf: Path) -> list[str]:
    text = command(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"])
    pages = text.split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    return [re.sub(r"\s+", " ", unicodedata.normalize("NFKC", page)).strip()
            for page in pages]


def inspect_pdf(pdf: Path, forbidden_terms: list[str]) -> tuple[dict, list[str]]:
    raw_info = command(["pdfinfo", str(pdf)])
    info = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in raw_info.splitlines() if ":" in line}
    pages = int(info.get("Pages", "0"))
    require(1 <= pages <= 14, f"Page count must be between 1 and 14: {pages}")
    require(info.get("Author", "") == "", f"PDF Author metadata is not empty: {info.get('Author')}")
    require(re.search(r"612(?:\.0+)? x 792(?:\.0+)? pts", info.get("Page size", "")) is not None,
            f"Unexpected page size: {info.get('Page size')}")
    require(pdf.stat().st_size <= 50_000_000, "PDF exceeds 50 MB")
    fonts = []
    for line in command(["pdffonts", str(pdf)]).splitlines()[2:]:
        if not line.strip():
            continue
        fields = re.match(r"(\S+)\s{2,}(.+?)\s{2,}(\S+)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$",
                          line.strip())
        require(fields is not None, f"Cannot parse pdffonts row: {line}")
        name, kind, encoding, embedded, subset, unicode = fields.groups()
        require(embedded == "yes", f"Font is not embedded: {name}")
        require(re.sub(r"\s", "", kind).lower() != "type3", f"Type 3 font is prohibited: {name}")
        fonts.append({"name": name, "type": kind, "encoding": encoding,
                      "embedded": embedded, "subset": subset, "unicode": unicode})
    require(fonts, "No fonts found in compiled PDF")
    require(any(re.search(r"NimbusRom|Times", f["name"], re.IGNORECASE) for f in fonts),
            "Expected Times-family manuscript fonts are absent")
    texts = page_text(pdf)
    require(len(texts) == pages, "PDF page-text extraction disagrees with page count")
    combined = "\n".join(texts)
    require(not UNFINISHED.search(combined), "Rendered PDF contains an unfinished-result marker")
    require("??" not in combined, "Rendered PDF contains unresolved reference marks")
    require(not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", combined),
            "Rendered PDF contains an email address")
    require(not re.search(r"\bAcknowledg(?:ment|ement|ments|ements)\b", combined, re.IGNORECASE),
            "Rendered review PDF contains an acknowledgments section")
    for term in forbidden_terms:
        require(term.casefold() not in combined.casefold(),
                f"Rendered PDF contains a project identifier: {term}")
    return {"pages": pages, "bytes": pdf.stat().st_size, "sha256": sha256(pdf),
            "page_size": info["Page size"], "pdf_author_empty": True,
            "fonts": fonts, "all_fonts_embedded": True, "type3_font_count": 0,
            "page_text_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in texts]}, texts


def compile_source(engine: Path, source: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
        env.pop(name, None)
    log = command([str(engine), "--only-cached", "--reruns", "3", "--keep-logs",
                   "--keep-intermediates", "--outdir", str(output), "main.tex"], cwd=source, env=env)
    (output / "engine_stdout.txt").write_text(log)
    final_log = (output / "main.log").read_text(errors="replace")
    for pattern in TEX_FAILURES:
        match = re.search(pattern, final_log, re.MULTILINE | re.IGNORECASE)
        require(not match, f"Final TeX log failed QA: {match.group(0) if match else pattern}")
    return output / "main.pdf"


def make_archive(source: Path, names: list[str], archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, (source / name).read_bytes())
    require(archive.stat().st_size <= 50_000_000, "Source archive exceeds 50 MB")


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".cvm-release-tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--expected-citations", type=int, default=51)
    parser.add_argument("--render-dpi", type=int, default=120)
    parser.add_argument("--check-source-only", action="store_true",
                        help="Read-only source validation; fail while manuscript contains pending results")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    paper = repo / PAPER_REL
    manifest = source_manifest(paper, args.expected_citations)
    forbidden = anonymity_terms(repo)
    for name in manifest["files"]:
        if name.endswith((".tex", ".bib", ".md")):
            text = (paper / name).read_text().casefold()
            require(not any(term.casefold() in text for term in forbidden),
                    f"Source archive member contains a project identifier: {name}")
    if args.check_source_only:
        print(json.dumps({"source_checks": "pass", **manifest}, indent=2))
        return 0
    evidence = release_evidence(repo)
    engine = args.engine.resolve()
    require(engine.is_file() and os.access(engine, os.X_OK), f"Tectonic is unavailable: {engine}")
    for name in ("pdfinfo", "pdffonts", "pdftotext", "pdftoppm"):
        require(shutil.which(name) is not None, f"Required tool is unavailable: {name}")

    stage = Path(tempfile.mkdtemp(prefix="cvm2027-release-"))
    print(f"Staging checks in {stage}; canonical artifacts remain unchanged until all checks pass.", flush=True)
    snapshot = stage / "source"
    snapshot.mkdir()
    for name in manifest["files"]:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paper / name, destination)
    pdf = compile_source(engine, snapshot, stage / "build")
    pdf_qa, texts = inspect_pdf(pdf, forbidden)
    aux = (stage / "build/main.aux").read_text()
    require(len(re.findall(r"\\bibcite\{", aux)) == args.expected_citations,
            "Compiled bibliography count disagrees with actual citation count")
    archive = stage / "cvm2027_review_source_20261026.zip"
    make_archive(snapshot, manifest["files"], archive)
    clean = stage / "clean-source"
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "Source archive CRC verification failed")
        require(sorted(zipped.namelist()) == manifest["files"], "Archive contains unexpected members")
        zipped.extractall(clean)
    for name in manifest["files"]:
        require(sha256(clean / name) == manifest["sha256"][name], f"Extracted source changed: {name}")
    rebuilt = compile_source(engine, clean, stage / "clean-build")
    rebuilt_qa, rebuilt_texts = inspect_pdf(rebuilt, forbidden)
    require(texts == rebuilt_texts, "Clean independent build has different page text or page count")
    images = stage / "pages"
    images.mkdir()
    command(["pdftoppm", "-r", str(args.render_dpi), "-png", str(pdf), str(images / "page")])
    rendered = sorted(images.glob("page-*.png"))
    require(len(rendered) == pdf_qa["pages"], "Not every PDF page rendered successfully")
    for name in manifest["files"]:
        require(sha256(paper / name) == manifest["sha256"][name],
                f"Source changed during build; refusing to publish stale artifacts: {name}")
    require(release_evidence(repo) == evidence,
            "Experiment provenance changed during the build; refusing to publish stale artifacts")

    targets = {
        "manuscript_pdf": paper / "main.pdf",
        "review_pdf": repo / "output/pdf/cvm2027_review_draft_20261026.pdf",
        "source_zip": repo / "output/source/cvm2027_review_source_20261026.zip",
        "manuscript_source_zip": paper / "cvm2027_review_source.zip",
        "qa_json": repo / "reports/cvm2027_build_qa_20261026.json",
        "qa_report": repo / "reports/cvm2027_build_qa_20261026.md",
        "page_images": repo / "output/qa/cvm2027_review_20261026/pages",
    }
    atomic_copy(pdf, targets["manuscript_pdf"])
    atomic_copy(pdf, targets["review_pdf"])
    atomic_copy(archive, targets["source_zip"])
    atomic_copy(archive, targets["manuscript_source_zip"])
    targets["page_images"].mkdir(parents=True, exist_ok=True)
    for image in rendered:
        atomic_copy(image, targets["page_images"] / image.name)
    for previous in targets["page_images"].glob("page-*.png"):
        if previous.name not in {image.name for image in rendered}:
            previous.unlink()
    qa = {
        "machine_checks": "pass", "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": {"path": str(engine), "sha256": sha256(engine),
                   "version": command([str(engine), "--version"]).strip(),
                   "flags": ["--only-cached", "--reruns", "3", "--keep-logs", "--keep-intermediates"]},
        "pdf": pdf_qa, "source_manifest": manifest, "experiment_evidence": evidence,
        "source_zip_sha256": sha256(archive), "source_zip_bytes": archive.stat().st_size,
        "independent_rebuild": {"pass": True, "page_text_identical": True,
                                "pages": rebuilt_qa["pages"], "pdf_sha256": rebuilt_qa["sha256"]},
        "visual_qa": {"status": "all_pages_rendered_awaiting_visual_review",
                      "render_dpi": args.render_dpi, "page_count": len(rendered),
                      "directory": str(targets["page_images"])},
        "artifacts": {key: str(path) for key, path in targets.items()},
        "staging_directory": str(stage), "submission_performed": False,
    }
    qa_stage = stage / "qa.json"
    qa_stage.write_text(json.dumps(qa, indent=2) + "\n")
    atomic_copy(qa_stage, targets["qa_json"])
    report = ["# CVM 2027 review-draft build QA — 2026-10-26", "",
              f"Machine checks passed. {pdf_qa['pages']} US Letter pages; {manifest['citation_count']} actual citations; {manifest['figure_count']} used figure assets.", "",
              "The full-split experiment report is complete with eight shared-attention cells and all twelve planned paired comparisons. The separate request-quality/cost report is complete with ten configurations and 100 questions per task. Their referenced raw-file hashes were verified before and after building.", "",
              "All PDF fonts are embedded, no Type 3 fonts are present, author metadata is empty, and the final TeX log has no undefined references, overfull boxes, or missing characters. Rendered content and uncommented manuscript TeX contain no unfinished-result markers. The anonymous source whitelist and rendered PDF contain no checked project identifiers.", "",
              "The minimal source ZIP was extracted into a separate clean directory and compiled independently. Page count and normalized text on every page match the release PDF. The archive includes only the source whitelist; datasets, weights, predictions, reports, build logs and compiled PDFs are excluded.", "",
              f"Every page was rendered at {args.render_dpi} dpi into `{targets['page_images']}`. Visual inspection remains a separate check; rendering does not certify appearance.", "",
              f"PDF SHA256: `{pdf_qa['sha256']}`", f"Source ZIP SHA256: `{sha256(archive)}`", "",
              f"Detailed machine-readable record: `{targets['qa_json']}`.",
              "This script prepared local review artifacts and did not upload or submit the paper."]
    report_stage = stage / "qa.md"
    report_stage.write_text("\n".join(report) + "\n")
    atomic_copy(report_stage, targets["qa_report"])
    print(json.dumps({"machine_checks": "pass", "pages": pdf_qa["pages"],
                      "artifacts": qa["artifacts"], "visual_review_required": True}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseError, OSError, ValueError, KeyError) as error:
        print(f"Release aborted: {error}", file=sys.stderr)
        raise SystemExit(1)
