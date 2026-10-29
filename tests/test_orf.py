import pandas as pd
import pysam
import pytest

pytest.importorskip("Bio")
from irtriage.orf import find_orfipy, run_orf_step  # noqa: E402

try:
    find_orfipy()
except RuntimeError:
    pytest.skip("orfipy not installed", allow_module_level=True)

PAD = "C" * 10
E1 = "GG" + "ATG" + "GCC" * 5 + "AA"  # ATG in frame; codon 8 = AA + first intron base
INTRON = "GTAAGT" + "C" * 30 + "TTTCAG"  # GTA|AGT...: retained intron reads AAG TAA -> stop inside the intron
E2 = "A" + "GGG" + "CCC" * 4 + "TAA" + "GG"
INTRON2 = "GTAAGT" + "C" * 10 + "CAG"
E3 = "C" * 80


def _revcomp(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def _build(tmp_path, strand):
    plus = PAD + E1 + INTRON + E2 + INTRON2 + E3 + PAD
    e1 = (len(PAD), len(PAD) + len(E1))
    i1 = (e1[1], e1[1] + len(INTRON))
    e2 = (i1[1], i1[1] + len(E2))
    i2 = (e2[1], e2[1] + len(INTRON2))
    e3 = (i2[1], i2[1] + len(E3))
    exons = [e1, e2, e3]
    genome = plus
    if strand == "-":
        n = len(plus)
        genome = _revcomp(plus)
        exons = [(n - b, n - a) for a, b in exons]
        i1 = (n - i1[1], n - i1[0])
    fa = tmp_path / "g.fa"
    fa.write_text(">chr1\n" + genome + "\n")
    pysam.faidx(str(fa))
    gtf = tmp_path / "g.gtf"
    gtf.write_text(
        "".join(
            f'chr1\ttest\texon\t{a + 1}\t{b}\t.\t{strand}\t.\tgene_id "G"; transcript_id "TX1";\n'
            for a, b in exons
        )
    )
    scored = pd.DataFrame(
        [dict(event_id="G|TX1|chr1:x", gene_id="G", transcript_id="TX1", chrom="chr1",
              start=i1[0], end=i1[1], strand=strand, call="real_IR"),
         dict(event_id="G|TX1|chr1:y", gene_id="G", transcript_id="TX1", chrom="chr1",
              start=i1[0] + 7, end=i1[1], strand=strand, call="real_IR"),  # not an annotated intron
         dict(event_id="G|TX1|chr1:z", gene_id="G", transcript_id="TX1", chrom="chr1",
              start=i1[0], end=i1[1], strand=strand, call="artifact")]
    )
    return scored, gtf, fa


@pytest.mark.parametrize("strand", ["+", "-"])
def test_longest_orf_with_ptc_in_intron(tmp_path, strand):
    scored, gtf, fa = _build(tmp_path, strand)
    out = tmp_path / "orf"
    t = run_orf_step(scored, gtf, fa, out, min_aa=5).set_index("event_id")

    assert len(t) == 2  # the artifact-call event is skipped
    assert t.loc["G|TX1|chr1:y", "orf_status"] == "the_intron_is_not_an_annotated_intron_of_this_transcript"

    r = t.loc["G|TX1|chr1:x"]
    assert r.orf_status == "ok"
    assert r.ir_orf_aa == 7 and r.ref_orf_aa == 12  # MAAAAAK vs MAAAAAKGPPPP
    assert bool(r.stop_in_intron) and bool(r.orf_overlaps_intron)
    assert bool(r.nmd_candidate)  # PTC well upstream of the last exon-exon junction
    assert not bool(r.protein_identical_to_ref)
    assert r.shared_nterm_aa == 7 and r.ir_novel_tail_aa == 0

    ir = (out / "ir_proteins.faa").read_text().split("\n")
    assert ir[1] == "MAAAAAK"
    assert (out / "ref_proteins.faa").read_text().split("\n")[1] == "MAAAAAKGPPPP"
