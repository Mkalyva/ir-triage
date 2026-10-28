"""Separate real intron retention from artifacts.

Two modes:

* weak supervision (default, no labels needed): an interpretable rule score gives
  pseudo-labels, a logistic regression is cross-fitted on them, and the final
  score blends the two;
* supervised (``labels=``): a random forest trained on your own real/artifact calls
  (e.g. RT-PCR or short-read validated events), with cross-validated AUC reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "log_n_retained",
    "ir_ratio_reads",
    "frac_retained_both_flank_spliced",
    "frac_retained_any_flank_spliced",
    "partial_fraction",
    "frac_alt_junction",
    "depth_ratio",
    "intron_depth_cv",
    "frac_intron_covered",
    "mean_mapq_retained",
    "nm_rate_delta",
    "strand_discordance",
    "priming_burden",
    "repeat_frac",
    "simple_repeat_frac",
    "log_intron_len",
]
WEIGHTS = dict(
    support=0.20,
    continuity=0.20,
    uniformity=0.10,
    flank_splicing=0.20,
    priming=0.10,
    strand=0.10,
    repeat=0.05,
    mapq=0.05,
)
HARD_FLAGS = {
    "internal_priming_like",
    "strand_discordant",
    "retained_reads_unspliced_context",
    "discontinuous_intron_coverage",
}
REAL_CUTOFF, ARTIFACT_CUTOFF = 0.7, 0.4


def derive(df):
    """Add model-ready columns to the raw event table."""
    d = df.copy()
    nonspliced = d.n_retained + d.n_left_partial + d.n_right_partial + d.n_internal
    d["log_n_retained"] = np.log1p(d.n_retained)
    d["log_intron_len"] = np.log10(d.intron_len)
    d["partial_fraction"] = (d.n_left_partial + d.n_right_partial + d.n_internal) / nonspliced.replace(0, np.nan)
    d["frac_alt_junction"] = d.n_alt_junction / (d.n_alt_junction + d.n_retained + d.n_spliced).replace(0, np.nan)
    d["depth_ratio"] = d.depth_ratio.clip(0, 2)
    d["nm_rate_delta"] = d.nm_rate_retained - d.nm_rate_spliced
    d["strand_discordance"] = d.frac_sense_spliced - d.frac_sense_retained
    d["priming_burden"] = d.n_intronic_3end_arich / (d.n_intronic_3end_arich + d.n_retained).replace(0, np.nan)
    return d


def rule_components(d):
    c = {}
    c["support"] = 1 - np.exp(-d.n_retained / 3)
    c["continuity"] = ((d.frac_intron_covered.fillna(0) - 0.6) / 0.4).clip(0, 1)
    c["uniformity"] = 1 - (d.intron_depth_cv.fillna(2) / 1.5).clip(0, 1)
    # where the gene shows no splicing elsewhere, flank-splicing is uninformative
    c["flank_splicing"] = pd.Series(
        np.where(d.flank_splice_informative == 0, 0.5, d.frac_retained_both_flank_spliced.fillna(0.5)),
        index=d.index,
    )
    # no fasta -> neutral; intronic 3' ends with A-rich genome downstream -> internal priming
    c["priming"] = (1 - d.priming_burden).where(d.n_intronic_3end_arich.notna(), 0.5).fillna(1.0)
    c["strand"] = 1 - (d.strand_discordance.fillna(0.25) / 0.5).clip(0, 1)
    c["repeat"] = 1 - d.simple_repeat_frac.fillna(0)
    c["mapq"] = (d.mean_mapq_retained.fillna(0) / 30).clip(0, 1)
    return pd.DataFrame(c)


def rule_score(d):
    comp = rule_components(d)
    return sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)


def flag_events(d):
    out = []
    for r in d.itertuples():
        f = []
        if r.n_retained == 0:
            f.append("no_full_read_support")
            if r.n_alt_junction >= 2:
                f.append("alt_junction_support")
            if r.n_left_partial + r.n_right_partial + r.n_internal >= 2:
                f.append("partial_only")
        elif r.n_retained < 3:
            f.append("low_read_support")
        if r.n_retained > 0:
            if r.frac_intron_covered < 0.8:
                f.append("discontinuous_intron_coverage")
            if r.flank_splice_informative == 1 and r.frac_retained_both_flank_spliced < 0.2:
                f.append("retained_reads_unspliced_context")
        if r.priming_burden >= 0.5:
            f.append("internal_priming_like")
        if r.strand_discordance > 0.4:
            f.append("strand_discordant")
        if r.simple_repeat_frac >= 0.5:
            f.append("simple_repeat_intron")
        out.append(";".join(f))
    return pd.Series(out, index=d.index)


def _weak_labels(d, rule, flags):
    y = np.full(len(d), -1)
    hard = flags.apply(lambda s: bool(HARD_FLAGS & set(s.split(";"))))
    y[((rule <= 0.35) | hard).to_numpy()] = 0
    y[((rule >= 0.75) & ~hard & (d.n_retained >= 3)).to_numpy()] = 1
    return y


def _logreg():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000),
    )


def _forest():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0),
    )


def _crossfit(X, y, make_model, seed=0):
    """Out-of-fold probabilities for labelled rows, fold-averaged for unlabelled rows."""
    lab, unl = np.where(y >= 0)[0], np.where(y < 0)[0]
    k = int(min(5, np.bincount(y[lab]).min()))
    pred, pred_unl = np.zeros(len(X)), np.zeros(len(unl))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X[lab], y[lab]):
        m = make_model().fit(X[lab[tr]], y[lab[tr]])
        pred[lab[te]] = m.predict_proba(X[lab[te]])[:, 1]
        if len(unl):
            pred_unl += m.predict_proba(X[unl])[:, 1] / k
    pred[unl] = pred_unl
    return pred


def score_events(events, labels=None, min_per_class=5):
    """Score an event table from ``irtriage.features.extract``.

    ``labels``: optional DataFrame with ``event_id`` and ``label`` (real/artifact or 1/0).
    Returns (scored table, info dict).
    """
    d = derive(events.reset_index(drop=True))
    d["rule_score"] = rule_score(d)
    d["qc_flags"] = flag_events(d)
    X = d[FEATURES].to_numpy(dtype=float)
    info = {"mode": "rules_only", "n_train_real": 0, "n_train_artifact": 0, "cv_auc": None}

    testable = (d.n_retained > 0).to_numpy()  # events with no full-length read are not scored by the model
    d["model_score"] = np.nan
    d["ai_score"] = d.rule_score

    if labels is not None:
        lab = labels.assign(
            y=labels["label"].astype(str).str.lower().map({"real": 1, "1": 1, "artifact": 0, "0": 0})
        ).dropna(subset=["y"])
        y = d.event_id.map(dict(zip(lab.event_id, lab.y))).fillna(-1).astype(int).to_numpy().copy()
        y[~testable] = -1
        info["mode"] = "supervised"
    else:
        y = _weak_labels(d, d.rule_score, d.qc_flags)
        y[~testable] = -1
        info["mode"] = "weak_supervision"

    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    info.update(n_train_real=n1, n_train_artifact=n0)
    if min(n1, n0) >= min_per_class:
        idx = np.where(testable)[0]
        model_prob = _crossfit(X[idx], y[idx], _forest if labels is not None else _logreg)
        d.loc[d.index[idx], "model_score"] = model_prob
        if labels is not None:
            lab_i = y[idx] >= 0
            info["cv_auc"] = float(roc_auc_score(y[idx][lab_i], model_prob[lab_i]))
            d.loc[d.index[idx], "ai_score"] = model_prob
        else:
            d.loc[d.index[idx], "ai_score"] = 0.5 * d.rule_score.iloc[idx] + 0.5 * model_prob
    else:
        info["mode"] = "rules_only"
        info["note"] = f"fewer than {min_per_class} events per class; using the rule score alone"

    call = np.where(d.ai_score >= REAL_CUTOFF, "real_IR", np.where(d.ai_score <= ARTIFACT_CUTOFF, "artifact", "uncertain"))
    d["call"] = np.where(testable, call, "no_full_length_evidence")
    return d, info
