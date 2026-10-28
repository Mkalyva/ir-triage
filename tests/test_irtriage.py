import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import synthetic  # noqa: E402

from irtriage import extract, score_events  # noqa: E402
from irtriage.features import RepeatIndex, classify_read  # noqa: E402

S, E, F, T = 1000, 2000, 20, 5


@pytest.mark.parametrize(
    "blocks,juncs,expected",
    [
        ([(100, 300), (500, 1000), (2000, 2300)], [(300, 500), (1000, 2000)], "spliced"),
        ([(500, 2300)], [], "retained"),
        ([(500, 1300)], [], "left_partial"),
        ([(1700, 2300)], [], "right_partial"),
        ([(1200, 1500)], [], "internal"),
        ([(500, 1000), (1400, 2300)], [(1000, 1400)], "alt_junction"),
        ([(100, 300), (2600, 2800)], [(300, 2600)], "other"),
    ],
)
def test_classify_read(blocks, juncs, expected):
    assert classify_read(blocks, juncs, S, E, F, T) == expected


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("irtriage")
    return synthetic.build(d), d


def test_extract_counts(data):
    (bam, fasta, table), _ = data
    events, reads = extract(str(bam), str(table), str(fasta))
    a = events[events.gene_id == "GENE_A"].iloc[0]
    b = events[events.gene_id == "GENE_B"].iloc[0]
    assert (a.n_retained, a.n_spliced) == (8, 12)
    assert a.frac_retained_both_flank_spliced == 1.0
    assert a.frac_intron_covered == 1.0
    assert (b.n_retained, b.n_left_partial) == (0, 6)
    assert b.n_intronic_3end_arich == 6
    assert set(reads.read_class) >= {"spliced", "retained", "left_partial"}


def test_scoring_calls(data):
    (bam, fasta, table), _ = data
    events, _ = extract(str(bam), str(table), str(fasta))
    scored, info = score_events(events)
    calls = dict(zip(scored.gene_id, scored.call))
    assert calls["GENE_A"] == "real_IR"
    assert calls["GENE_B"] == "no_full_length_evidence"
    assert "internal_priming_like" in scored.set_index("gene_id").loc["GENE_B", "qc_flags"]
    assert info["mode"] == "rules_only"  # two events is too few to train a model


def test_repeat_index(tmp_path):
    bed = tmp_path / "rmsk.bed"
    bed.write_text("chr1\t1000\t1300\tAluSx\nchr1\t1500\t1600\t(CA)n\nchr1\t1100\t1200\tL1PA2\n")
    frac, simple, top = RepeatIndex(str(bed)).summarize("chr1", 1000, 2000)
    assert frac == pytest.approx(0.4)
    assert simple == pytest.approx(0.1)
    assert top == "AluSx"


def test_model_paths(data):
    import numpy as np
    import pandas as pd

    (bam, fasta, table), _ = data
    events, _ = extract(str(bam), str(table), str(fasta))
    base = events[events.gene_id == "GENE_A"].iloc[0]
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        r = base.copy()
        r["event_id"] = f"e{i}"
        r["n_retained"] = int(rng.integers(3, 15))
        if i % 2:  # artifact-like: unspliced context, ragged coverage, A-primed ends, discordant strand
            r["frac_retained_both_flank_spliced"] = rng.uniform(0, 0.1)
            r["frac_intron_covered"] = rng.uniform(0.3, 0.7)
            r["intron_depth_cv"] = rng.uniform(1, 2)
            r["n_intronic_3end_arich"] = int(rng.integers(5, 15))
            r["frac_sense_retained"] = 0.4
        rows.append(r)
    big = pd.DataFrame(rows).reset_index(drop=True)
    scored, info = score_events(big)
    assert info["mode"] == "weak_supervision"
    assert scored.loc[::2, "ai_score"].mean() > 0.7
    assert scored.loc[1::2, "ai_score"].mean() < 0.4

    labels = pd.DataFrame({"event_id": big.event_id, "label": ["artifact" if i % 2 else "real" for i in range(60)]})
    _, info = score_events(big, labels)
    assert info["mode"] == "supervised" and info["cv_auc"] > 0.9
