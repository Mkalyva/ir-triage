"""Tiny synthetic dataset: one real-looking retained intron and one internal-priming artifact."""
import random

import pysam

# gene A exons: 100-300, 500-1000, 2000-2300, 2600-2800 ; target intron 1000-2000 (1-based 1001-2000)
# gene B exons: 4000-4300, 4500-5500, 6500-6800, 7000-7300 ; target intron 5500-6500 (1-based 5501-6500)
TABLE = (
    "gene_id\ttranscript_id\tchrom\tstart\tend\tstrand\n"
    "GENE_A\tTX_A\tchr1\t1001\t2000\t+\n"
    "GENE_B\tTX_B\tchr1\t5501\t6500\t+\n"
)


def _read(header, name, start, cigar):
    r = pysam.AlignedSegment(header)
    r.query_name, r.reference_id, r.reference_start = name, 0, start
    r.mapping_quality, r.flag = 60, 0
    r.cigarstring = cigar
    qlen = sum(n for op, n in r.cigartuples if op in (0, 1, 4))
    r.query_sequence = "A" * qlen
    r.set_tag("NM", 0)
    return r


def build(outdir):
    rng = random.Random(1)
    genome = [rng.choice("ACG") for _ in range(10000)]  # no T/A bias
    genome[5800:5820] = "A" * 20  # A-rich stretch where the artifact reads end
    fasta = outdir / "ref.fa"
    fasta.write_text(">chr1\n" + "".join(genome) + "\n")
    pysam.faidx(str(fasta))

    header = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10000}]})
    reads = []
    for i in range(12):  # spliced across target intron A
        reads.append(_read(header, f"A_spl{i}", 100, "200M200N500M1000N300M300N200M"))
    for i in range(8):  # retained intron A, spliced on both sides
        reads.append(_read(header, f"A_ret{i}", 100, "200M200N1800M300N200M"))
    for i in range(15):  # spliced across target intron B
        reads.append(_read(header, f"B_spl{i}", 4000, "300M200N1000M1000N300M200N300M"))
    for i in range(6):  # reads that end inside intron B at the A-rich stretch
        reads.append(_read(header, f"B_part{i}", 4000, "300M200N1300M"))

    unsorted, bam = outdir / "u.bam", outdir / "test.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for r in reads:
            out.write(r)
    pysam.sort("-o", str(bam), str(unsorted))
    pysam.index(str(bam))
    table = outdir / "table.tsv"
    table.write_text(TABLE)
    return bam, fasta, table
