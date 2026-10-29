"""Retained-intron transcripts -> longest ORF (orfipy + Biopython) -> protein FASTA for AlphaFold."""
from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pysam
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

STOPS = {"TAA", "TAG", "TGA"}
NMD_RULE_NT = 50  # a stop >50 nt upstream of the last exon-exon junction triggers NMD
HEADER_RE = re.compile(r"\[(\d+)-(\d+)\]\(([+-])\)\s+type:(\S+)\s+length:(\d+)\s+frame:(-?\d+)\s+start:(\w+)\s+stop:(\w+)")


@dataclasses.dataclass
class Orf:
    start: int  # 0-based in the transcript (sense) orientation
    end: int  # exclusive, stop codon not included
    orf_type: str
    start_codon: str
    has_stop: bool
    protein: str

    @property
    def nt_len(self):
        return self.end - self.start

    @property
    def stop_end(self):
        return self.end + 3 if self.has_stop else self.end


def load_exons(gtf_path, transcript_ids):
    """Exons (0-based half-open, genomic order) for the requested transcripts from a GTF."""
    wanted, out = set(transcript_ids), {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            m = re.search(r'transcript_id "([^"]+)"', f[8])
            if not m or m.group(1) not in wanted:
                continue
            tx = out.setdefault(m.group(1), dict(chrom=f[0], strand=f[6], exons=[]))
            tx["exons"].append((int(f[3]) - 1, int(f[4])))
    for tx in out.values():
        tx["exons"].sort()
    return out


def _revcomp(seq):
    return str(Seq(seq).reverse_complement())


def build_transcripts(tx, intron_start, intron_end, genome, tol=0):
    """Spliced (reference) and intron-retained sequences, both in sense orientation.

    Returns a dict with ref_seq, ir_seq, the intron span in IR-transcript coordinates
    and the exon-exon junction positions of the IR transcript.
    """
    exons, strand = tx["exons"], tx["strand"]
    idx = next(
        (
            i
            for i in range(len(exons) - 1)
            if abs(exons[i][1] - intron_start) <= tol and abs(exons[i + 1][0] - intron_end) <= tol
        ),
        None,
    )
    if idx is None:
        raise ValueError("the intron is not an annotated intron of this transcript")
    ir_exons = exons[:idx] + [(exons[idx][0], exons[idx + 1][1])] + exons[idx + 2 :]

    def seq_of(exs):
        return "".join(genome.fetch(tx["chrom"], a, b).upper() for a, b in exs)

    ref_seq, ir_seq = seq_of(exons), seq_of(ir_exons)
    total = len(ir_seq)
    lo = sum(b - a for a, b in ir_exons[:idx]) + (exons[idx][1] - exons[idx][0])
    hi = lo + (exons[idx + 1][0] - exons[idx][1])
    junctions, pos = [], 0
    for a, b in ir_exons[:-1]:
        pos += b - a
        junctions.append(pos)
    if strand == "-":
        ref_seq, ir_seq = _revcomp(ref_seq), _revcomp(ir_seq)
        lo, hi = total - hi, total - lo
        junctions = sorted(total - j for j in junctions)
    return dict(ref_seq=ref_seq, ir_seq=ir_seq, intron_span=(lo, hi), junctions=junctions)


def find_orfipy():
    exe = Path(sys.executable).parent / "orfipy"
    if exe.exists():
        return str(exe)
    found = shutil.which("orfipy")
    if not found:
        raise RuntimeError("orfipy not found; install with: pip install 'irtriage[orf]'")
    return found


def run_orfipy(fasta, outdir, min_aa=30, start_codons="ATG", allow_partial=False, procs=None):
    """Run orfipy on sense-strand sequences; returns the path of the ORF DNA FASTA."""
    cmd = [
        find_orfipy(), str(fasta),
        "--strand", "f",
        "--start", start_codons,
        "--min", str(min_aa * 3),
        "--dna", "orfs.dna.fa",
        "--bed", "orfs.bed",
        "--outdir", str(outdir),
    ]
    if allow_partial:
        cmd.append("--partial-3")
    if procs:
        cmd += ["--procs", str(procs)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(outdir) / "orfs.dna.fa"


def longest_orfs(orf_dna_fasta, seqs):
    """Pick the longest ORF per transcript with Biopython.

    ``seqs`` maps transcript name -> sense sequence (used to check for a stop codon).
    Ties go to complete ORFs, then to the earliest start.
    """
    best = {}
    for rec in SeqIO.parse(str(orf_dna_fasta), "fasta"):
        name = rec.id.rsplit("_ORF.", 1)[0]
        m = HEADER_RE.search(rec.description)
        start, end = int(m.group(1)), int(m.group(2))
        nt = rec.seq[: (len(rec.seq) // 3) * 3]
        has_stop = seqs[name][end : end + 3] in STOPS
        orf = Orf(start, end, m.group(4), m.group(7), has_stop, str(nt.translate(table=1)))
        key = (orf.nt_len, orf.has_stop, -orf.start)
        if name not in best or key > best[name][0]:
            best[name] = (key, orf)
    return {k: v[1] for k, v in best.items()}


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def annotate(ir_orf, ref_orf, built):
    """Compare the IR ORF with the reference ORF and the retained intron."""
    lo, hi = built["intron_span"]
    row = dict(intron_len=hi - lo, intron_in_frame=(hi - lo) % 3 == 0)
    if ir_orf is None:
        return dict(row, orf_status="no_orf")
    ref_prot = ref_orf.protein if ref_orf else ""
    prefix = _common_prefix(ir_orf.protein, ref_prot)
    stop_in_intron = ir_orf.has_stop and lo <= ir_orf.end < hi
    later = [j for j in built["junctions"] if j > ir_orf.stop_end]
    row.update(
        orf_status="ok",
        ir_orf_start=ir_orf.start,
        ir_orf_end=ir_orf.end,
        ir_orf_type=ir_orf.orf_type,
        ir_start_codon=ir_orf.start_codon,
        ir_orf_aa=len(ir_orf.protein),
        ref_orf_aa=len(ref_prot),
        orf_overlaps_intron=ir_orf.start < hi and ir_orf.stop_end > lo,
        orf_spans_intron=ir_orf.start < lo and ir_orf.stop_end > hi,
        stop_in_intron=stop_in_intron,
        nmd_candidate=bool(ir_orf.has_stop and later and (max(later) - ir_orf.stop_end) > NMD_RULE_NT),
        protein_identical_to_ref=ir_orf.protein == ref_prot,
        shared_nterm_aa=prefix,
        ir_novel_tail_aa=len(ir_orf.protein) - prefix,
    )
    return row


def run_orf_step(
    scored, gtf, genome_fasta, outdir, calls=("real_IR",), min_aa=30,
    start_codons="ATG", allow_partial=False, tol=0,
):
    """scored: events table from ``score_events`` (0-based start/end). Returns the ORF table."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    events = scored[scored.call.isin(calls)].reset_index(drop=True)
    txs = load_exons(gtf, events.transcript_id.unique())
    genome = pysam.FastaFile(str(genome_fasta))

    records, seqs, built_by, rows = [], {}, {}, []
    for i, ev in enumerate(events.itertuples()):
        base = dict(event_id=ev.event_id, gene_id=ev.gene_id, transcript_id=ev.transcript_id)
        tx = txs.get(ev.transcript_id)
        if tx is None:
            rows.append(dict(base, orf_status="transcript_not_in_gtf"))
            continue
        try:
            b = build_transcripts(tx, ev.start, ev.end, genome, tol)
        except ValueError as exc:
            rows.append(dict(base, orf_status=str(exc).replace(" ", "_")))
            continue
        tag = f"ev{i:05d}"
        for kind in ("ref", "ir"):
            seqs[f"{tag}_{kind}"] = b[f"{kind}_seq"]
            records.append(SeqRecord(Seq(b[f"{kind}_seq"]), id=f"{tag}_{kind}", description=""))
        built_by[tag] = (base, b)
    SeqIO.write(records, outdir / "transcripts.fa", "fasta")

    orfs = {}
    if records:
        orfs = longest_orfs(run_orfipy(outdir / "transcripts.fa", outdir, min_aa, start_codons, allow_partial), seqs)

    ir_prot, ref_prot = [], []
    for tag, (base, b) in built_by.items():
        ir, ref = orfs.get(f"{tag}_ir"), orfs.get(f"{tag}_ref")
        rows.append(dict(base, **annotate(ir, ref, b)))
        label = f"{base['gene_id']}_{base['transcript_id']}_{base['event_id'].rsplit('|', 1)[-1].replace(':', '_')}"
        if ir:
            ir_prot.append(SeqRecord(Seq(ir.protein), id=f"{label}_IR", description=f"longest_ORF {len(ir.protein)}aa"))
        if ref:
            ref_prot.append(SeqRecord(Seq(ref.protein), id=f"{label}_REF", description=f"longest_ORF {len(ref.protein)}aa"))
    SeqIO.write(ir_prot, outdir / "ir_proteins.faa", "fasta")
    SeqIO.write(ref_prot, outdir / "ref_proteins.faa", "fasta")

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "longest_orfs.tsv", sep="\t", index=False)
    return table
