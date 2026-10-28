# irtriage

Triage intron-retention (IR) events called from long-read RNA-seq. Given a table of IR
coordinates, `irtriage` goes back to the BAM with pysam, classifies every read at each
intron, adds repeat and internal-priming context, and scores each event as
`real_IR`, `artifact` or `uncertain`.

## Install

```bash
pip install -e .          # needs pysam, pandas, numpy, scikit-learn
```

## Input

**IR table** (TSV/CSV): `gene_id, transcript_id, chrom, start, end` and optional `strand`.
Coordinates are 1-based inclusive (GTF-style) by default; pass `--zero-based` for BED-style.

**BAM**: coordinate-sorted and indexed (e.g. minimap2 `-ax splice`). Optional:
`--fasta` (indexed genome; enables internal-priming detection) and `--repeats`
(RepeatMasker BED4). Build the repeat BED from the UCSC `rmsk` table:

```bash
zcat rmsk.txt.gz | awk -v OFS='\t' '{print $6,$7,$8,$12"|"$13"|"$11}' > rmsk.bed
```

## Usage

```bash
irtriage run --bam sample.bam --table ir_events.tsv --fasta genome.fa \
    --repeats rmsk.bed --reads-out reads.tsv -o ir_scored.tsv

# or in two steps (re-score without re-reading the BAM)
irtriage extract --bam sample.bam --table ir_events.tsv --fasta genome.fa -o evidence.tsv
irtriage score   --events evidence.tsv [--labels validated.tsv] -o ir_scored.tsv
```

## Read classes (per intron)

| class | meaning |
|---|---|
| `spliced` | an `N` gap matches the intron (± `--tol` bp) |
| `retained` | one contiguous alignment block covers the intron plus `--min-flank` bp of both flanking exons |
| `alt_junction` | a junction has one end inside the intron (cryptic splice site, i.e. partial IR) |
| `left_partial` / `right_partial` | read enters from one exon and ends inside the intron |
| `internal` | read lies wholly inside the intron |
| `other` | overlaps the window without informing on the intron (e.g. exon skipping) |

Only `retained` reads count as full-length IR evidence. Partial classes are reported
(`n_left_partial`, `n_alt_junction`, ...) but not counted as IR, because truncated reads,
pre-mRNA and internal poly-A priming look identical. Events with no `retained` read get
`call = no_full_length_evidence`; check `qc_flags` for `alt_junction_support` or
`partial_only` to follow those up separately.

## Features and flags

Per event: read counts per class, IR ratio (`retained / (retained + spliced)`),
fraction of retained reads that are also spliced upstream and downstream, intron vs
flank depth, coverage continuity and uniformity, MAPQ, mismatch rate relative to
spliced reads, strand discordance relative to spliced reads, intronic 3′ ends sitting
on A-rich genome (internal priming), and RepeatMasker overlap (`repeat_frac`,
`simple_repeat_frac`, `repeat_top`).

`qc_flags` lists human-readable reasons: `low_read_support`,
`discontinuous_intron_coverage`, `retained_reads_unspliced_context`,
`internal_priming_like`, `strand_discordant`, `simple_repeat_intron`.

## The scoring model

* **Rule score** (always computed): weighted mean of interpretable evidence components
  (read support, continuity, uniformity, flank splicing, priming, strand, repeats, MAPQ).
* **Default mode (weak supervision)**: confident rule-based pseudo-labels train a
  cross-fitted logistic regression; `ai_score = 0.5 * rule + 0.5 * model`. With fewer
  than 5 pseudo-labelled events per class it falls back to the rule score alone.
* **Supervised mode** (`--labels`, columns `event_id,label` with `real`/`artifact`):
  a random forest on your own validated calls; cross-validated AUC is printed.

Cut-offs: `real_IR` ≥ 0.7, `artifact` ≤ 0.4, otherwise `uncertain`.

**Limitations.** The weak-supervision model learns from labels derived from the same
features, so it smooths the rule score rather than adding independent evidence. Treat
it as a prioritisation aid. Weights and cut-offs are heuristics that have only been
tested on synthetic data. Validate on events you can confirm (RT-PCR, short-read IR
tools, matched direct-RNA vs cDNA) and use `--labels` once you have them.

## Tests

```bash
pip install -e ".[test]" && pytest
```
