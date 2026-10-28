"""Per-read evidence for intron-retention (IR) events, extracted from a BAM with pysam."""
from __future__ import annotations

import bisect
import dataclasses
import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd
import pysam

READ_CLASSES = (
    "spliced",
    "retained",
    "alt_junction",
    "left_partial",
    "right_partial",
    "internal",
    "other",
)
UNSPLICED_CLASSES = ("retained", "alt_junction", "left_partial", "right_partial", "internal")
SIMPLE_REPEAT_TAGS = ("simple_repeat", "low_complexity", "satellite")


@dataclasses.dataclass
class Params:
    min_flank: int = 20  # bp of aligned sequence required in each flanking exon for "retained"
    tol: int = 5  # bp tolerance when matching a read junction to the intron boundaries
    min_mapq: int = 1
    depth_window: int = 100  # bp of exon flank used as the depth baseline
    max_del: int = 50  # deletions longer than this break an alignment block
    arich_window: int = 20  # bp of genome inspected downstream of an intronic 3' end
    arich_threshold: float = 0.65


@dataclasses.dataclass
class Intron:
    event_id: str
    gene_id: str
    transcript_id: str
    chrom: str
    start: int  # 0-based, inclusive
    end: int  # 0-based, exclusive
    strand: str  # '+', '-' or '.'


def parse_alignment(read, max_del=50):
    """Return (blocks, junctions) as 0-based half-open reference intervals.

    Blocks are contiguous aligned segments (small deletions stay inside a block);
    junctions are the N gaps of the CIGAR string.
    """
    blocks, juncs = [], []
    pos = read.reference_start
    bstart = pos
    for op, ln in read.cigartuples:
        if op in (0, 7, 8):
            pos += ln
        elif op == 2:
            if ln > max_del:
                blocks.append((bstart, pos))
                pos += ln
                bstart = pos
            else:
                pos += ln
        elif op == 3:
            blocks.append((bstart, pos))
            juncs.append((pos, pos + ln))
            pos += ln
            bstart = pos
    blocks.append((bstart, pos))
    return [b for b in blocks if b[1] > b[0]], juncs


def _matches(junc, s, e, tol):
    return abs(junc[0] - s) <= tol and abs(junc[1] - e) <= tol


def classify_read(blocks, juncs, s, e, flank, tol):
    """Assign one alignment to a class relative to the intron [s, e)."""
    if any(_matches(j, s, e, tol) for j in juncs):
        return "spliced"
    if any(bs <= s - flank and be >= e + flank for bs, be in blocks):
        return "retained"
    for js, je in juncs:
        # a junction with an end inside the intron = cryptic splice site (partial IR)
        if s + tol < js < e - tol or s + tol < je < e - tol:
            return "alt_junction"
    touching = [(bs, be) for bs, be in blocks if be > s and bs < e]
    if not touching:
        return "other"
    left = any(bs <= s - flank and be > s for bs, be in touching)
    right = any(be >= e + flank and bs < e for bs, be in touching)
    if left and right:
        return "other"
    if left:
        return "left_partial"
    if right:
        return "right_partial"
    return "internal"


def _covered_bp(intervals):
    total, cur_s, cur_e = 0, None, None
    for a, b in sorted(intervals):
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _is_simple(name):
    low = name.lower()
    return any(t in low for t in SIMPLE_REPEAT_TAGS) or bool(re.fullmatch(r"\(.+\)n", name))


class RepeatIndex:
    """Interval lookup over a BED(4) of RepeatMasker annotations."""

    def __init__(self, bed_path):
        raw = {}
        with open(bed_path) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.rstrip("\n").split("\t")
                name = f[3] if len(f) > 3 else "repeat"
                raw.setdefault(f[0], []).append((int(f[1]), int(f[2]), name))
        self.chroms = {}
        for chrom, iv in raw.items():
            iv.sort()
            ends = [x[1] for x in iv]
            self.chroms[chrom] = (
                [x[0] for x in iv],
                ends,
                [x[2] for x in iv],
                list(itertools.accumulate(ends, max)),
            )

    def overlaps(self, chrom, s, e):
        if chrom not in self.chroms:
            return []
        starts, ends, names, pmax = self.chroms[chrom]
        i = bisect.bisect_left(starts, e) - 1
        hits = []
        while i >= 0 and pmax[i] > s:
            if ends[i] > s:
                hits.append((max(starts[i], s), min(ends[i], e), names[i]))
            i -= 1
        return hits

    def summarize(self, chrom, s, e):
        """Return (repeat_frac, simple_repeat_frac, top_repeat_name)."""
        hits = self.overlaps(chrom, s, e)
        if not hits:
            return 0.0, 0.0, ""
        length = e - s
        total = _covered_bp([(a, b) for a, b, _ in hits])
        simple = _covered_bp([(a, b) for a, b, n in hits if _is_simple(n)])
        bp = Counter()
        for a, b, n in hits:
            bp[n] += b - a
        return total / length, simple / length, bp.most_common(1)[0][0]


def _mean(a):
    return float(np.mean(a)) if len(a) else float("nan")


def extract_event(bam, intron, params, fasta=None, repeats=None):
    """Collect read-level evidence for one intron. Returns (features dict, read records)."""
    s, e, W = intron.start, intron.end, params.depth_window
    lo, hi = max(0, s - W), e + W
    diff = np.zeros(hi - lo + 1, dtype=np.int64)

    counts = Counter()
    retained_ends = set()
    ret = dict(mapq=[], nm=[], sense=[], both=0, any=0)
    spl = dict(nm=[], sense=[], other_junc=0)
    n3 = 0
    n3_arich = 0 if fasta is not None else float("nan")
    reads = []

    for read in bam.fetch(intron.chrom, lo, hi):
        if (
            read.is_unmapped
            or read.is_secondary
            or read.is_supplementary
            or read.mapping_quality < params.min_mapq
            or not read.cigartuples
        ):
            continue
        blocks, juncs = parse_alignment(read, params.max_del)
        for bs, be in blocks:
            a, b = max(bs, lo) - lo, min(be, hi) - lo
            if b > a:
                diff[a] += 1
                diff[b] -= 1

        cls = classify_read(blocks, juncs, s, e, params.min_flank, params.tol)
        counts[cls] += 1
        rstrand = "-" if read.is_reverse else "+"
        sense = (rstrand == intron.strand) if intron.strand in "+-" else None
        nm = read.get_tag("NM") / max(1, read.query_alignment_length) if read.has_tag("NM") else None

        n_left = sum(1 for js, je in juncs if je <= s + params.tol)
        n_right = sum(1 for js, je in juncs if js >= e - params.tol)

        if cls == "spliced":
            if nm is not None:
                spl["nm"].append(nm)
            if sense is not None:
                spl["sense"].append(sense)
            if any(not _matches(j, s, e, params.tol) for j in juncs):
                spl["other_junc"] += 1
        elif cls == "retained":
            ret["mapq"].append(read.mapping_quality)
            if nm is not None:
                ret["nm"].append(nm)
            if sense is not None:
                ret["sense"].append(sense)
            ret["both"] += n_left > 0 and n_right > 0
            ret["any"] += n_left > 0 or n_right > 0
            retained_ends.add((read.reference_start, read.reference_end))

        three_in, arich = False, False
        if cls in UNSPLICED_CLASSES:
            tx = intron.strand if intron.strand in "+-" else rstrand
            three = read.reference_end if tx == "+" else read.reference_start
            if s + params.tol <= three <= e - params.tol:
                three_in = True
                n3 += 1
                if fasta is not None:
                    w = params.arich_window
                    if tx == "+":
                        seq, base = fasta.fetch(intron.chrom, three, three + w).upper(), "A"
                    else:
                        seq, base = fasta.fetch(intron.chrom, max(0, three - w), three).upper(), "T"
                    if seq and seq.count(base) / len(seq) >= params.arich_threshold:
                        arich = True
                        n3_arich += 1

        reads.append(
            dict(
                event_id=intron.event_id,
                read_name=read.query_name,
                read_class=cls,
                mapq=read.mapping_quality,
                ref_start=read.reference_start,
                ref_end=read.reference_end,
                read_strand=rstrand,
                n_junctions_left=n_left,
                n_junctions_right=n_right,
                three_prime_in_intron=three_in,
                three_prime_polyA_like=arich,
            )
        )

    depth = np.cumsum(diff)[:-1]
    left = depth[max(0, s - W) - lo : s - lo]
    right = depth[e - lo : min(hi, e + W) - lo]
    intr = depth[s - lo : e - lo]
    flank_mean = np.nanmean([_mean(left), _mean(right)]) if (len(left) or len(right)) else float("nan")
    intr_mean = _mean(intr)

    n_ret, n_spl = counts["retained"], counts["spliced"]
    rep_frac = simple_frac = float("nan")
    rep_top = ""
    if repeats is not None:
        rep_frac, simple_frac, rep_top = repeats.summarize(intron.chrom, s, e)

    feats = dict(
        event_id=intron.event_id,
        gene_id=intron.gene_id,
        transcript_id=intron.transcript_id,
        chrom=intron.chrom,
        start=s,
        end=e,
        strand=intron.strand,
        intron_len=e - s,
        **{f"n_{c}": counts[c] for c in READ_CLASSES},
        n_retained_distinct=len(retained_ends),
        ir_ratio_reads=n_ret / (n_ret + n_spl) if (n_ret + n_spl) else float("nan"),
        frac_retained_both_flank_spliced=ret["both"] / n_ret if n_ret else float("nan"),
        frac_retained_any_flank_spliced=ret["any"] / n_ret if n_ret else float("nan"),
        flank_splice_informative=(
            float(spl["other_junc"] / n_spl >= 0.2) if n_spl >= 3 else float("nan")
        ),
        depth_left_flank=_mean(left),
        depth_right_flank=_mean(right),
        depth_intron_mean=intr_mean,
        depth_ratio=intr_mean / flank_mean if flank_mean and flank_mean > 0 else float("nan"),
        intron_depth_cv=float(intr.std() / intr_mean) if intr_mean and intr_mean > 0 else float("nan"),
        frac_intron_covered=float((intr > 0).mean()) if len(intr) else float("nan"),
        mean_mapq_retained=_mean(ret["mapq"]),
        nm_rate_retained=_mean(ret["nm"]),
        nm_rate_spliced=_mean(spl["nm"]),
        frac_sense_retained=_mean(ret["sense"]),
        frac_sense_spliced=_mean(spl["sense"]),
        n_intronic_3end=n3,
        n_intronic_3end_arich=n3_arich,
        repeat_frac=rep_frac,
        simple_repeat_frac=simple_frac,
        repeat_top=rep_top,
    )
    return feats, reads


def read_table(path, one_based=True):
    """Load the IR table: gene_id, transcript_id, chrom, start, end [, strand]."""
    df = pd.read_csv(path, sep=None, engine="python", dtype={"chrom": str})
    missing = {"gene_id", "transcript_id", "chrom", "start", "end"} - set(df.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    if "strand" not in df:
        df["strand"] = "."
    df["strand"] = df["strand"].fillna(".").astype(str)
    if one_based:  # GTF-style 1-based inclusive -> 0-based half-open
        df["start"] = df["start"].astype(int) - 1
    df["start"], df["end"] = df["start"].astype(int), df["end"].astype(int)
    if (df["end"] <= df["start"]).any():
        raise ValueError("every row needs end > start")
    return df


def extract(bam_path, table_path, fasta_path=None, repeats_path=None, params=None, one_based=True):
    """Run extraction for every row of the table. Returns (events_df, reads_df)."""
    params = params or Params()
    table = read_table(table_path, one_based)
    fasta = pysam.FastaFile(fasta_path) if fasta_path else None
    repeats = RepeatIndex(repeats_path) if repeats_path else None
    feats, reads = [], []
    with pysam.AlignmentFile(bam_path) as bam:
        known = set(bam.references)
        for row in table.itertuples(index=False):
            if row.chrom not in known:
                raise ValueError(f"chrom {row.chrom!r} not in BAM header")
            eid = f"{row.gene_id}|{row.transcript_id}|{row.chrom}:{row.start + 1}-{row.end}"
            intron = Intron(eid, row.gene_id, row.transcript_id, row.chrom, row.start, row.end, row.strand)
            f, r = extract_event(bam, intron, params, fasta, repeats)
            feats.append(f)
            reads.extend(r)
    return pd.DataFrame(feats), pd.DataFrame(reads)
