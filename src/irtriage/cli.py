from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .classify import score_events
from .features import Params, extract


def _add_extract_args(p):
    p.add_argument("--bam", required=True, help="coordinate-sorted, indexed long-read BAM")
    p.add_argument("--table", required=True, help="TSV/CSV: gene_id, transcript_id, chrom, start, end [, strand]")
    p.add_argument("--fasta", help="indexed genome FASTA (enables internal-priming detection)")
    p.add_argument("--repeats", help="RepeatMasker BED4 (chrom, start, end, name)")
    p.add_argument("--zero-based", action="store_true", help="table coords are BED-style 0-based half-open")
    p.add_argument("--reads-out", help="write per-read class assignments to this TSV")
    d = Params()
    p.add_argument("--min-flank", type=int, default=d.min_flank)
    p.add_argument("--tol", type=int, default=d.tol)
    p.add_argument("--min-mapq", type=int, default=d.min_mapq)


def _run_extract(a):
    params = Params(min_flank=a.min_flank, tol=a.tol, min_mapq=a.min_mapq)
    events, reads = extract(a.bam, a.table, a.fasta, a.repeats, params, one_based=not a.zero_based)
    if a.reads_out:
        reads.to_csv(a.reads_out, sep="\t", index=False)
    return events


def _write_scored(events, labels_path, out):
    labels = pd.read_csv(labels_path, sep=None, engine="python") if labels_path else None
    scored, info = score_events(events, labels)
    scored.to_csv(out, sep="\t", index=False)
    print(json.dumps(info), file=sys.stderr)
    print(scored.call.value_counts().to_string(), file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="irtriage", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="BAM + IR table -> per-event read evidence")
    _add_extract_args(p)
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("score", help="evidence table -> real/artifact calls")
    p.add_argument("--events", required=True)
    p.add_argument("--labels", help="TSV with event_id,label (real/artifact) for supervised mode")
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("run", help="extract + score")
    _add_extract_args(p)
    p.add_argument("--labels")
    p.add_argument("-o", "--out", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "extract":
        _run_extract(a).to_csv(a.out, sep="\t", index=False)
    elif a.cmd == "score":
        _write_scored(pd.read_csv(a.events, sep="\t"), a.labels, a.out)
    else:
        _write_scored(_run_extract(a), a.labels, a.out)


if __name__ == "__main__":
    main()
