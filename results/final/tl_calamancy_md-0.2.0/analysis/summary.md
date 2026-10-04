# Results analysis -- TEST set, tl_calamancy_md-0.2.0

## baseline (30 runs)

**Mean confusion matrix** (rows = true label, columns = predicted)

| True \\ Predicted | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 835.6 | 174.8 | 16.5 |
| Neutral | 184.0 | 803.3 | 99.6 |
| Positive | 21.1 | 146.0 | 865.9 |

**Per-class breakdown**

| Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Negative | 835.6 | 205.1 | 191.4 | 1914.9 | 0.8029 | 0.8137 | 0.8083 |
| Neutral | 803.3 | 320.8 | 283.7 | 1739.2 | 0.7146 | 0.7390 | 0.7266 |
| Positive | 865.9 | 116.2 | 167.1 | 1997.8 | 0.8817 | 0.8383 | 0.8594 |

**Overall metrics** (mean ± sd)

| Metric | Mean | SD |
|---|---|---|
| accuracy | 0.7960 | 0.0051 |
| macro_averaged_accuracy | 0.8640 | 0.0034 |
| macro_precision | 0.8014 | 0.0063 |
| macro_recall | 0.7970 | 0.0050 |
| macro_f1 | 0.7979 | 0.0055 |
| weighted_precision | 0.8000 | 0.0060 |
| weighted_recall | 0.7960 | 0.0051 |
| weighted_f1 | 0.7966 | 0.0056 |

## proposed-projection (30 runs)

**Mean confusion matrix** (rows = true label, columns = predicted)

| True \\ Predicted | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 836.1 | 174.3 | 16.6 |
| Neutral | 182.1 | 802.9 | 101.9 |
| Positive | 18.9 | 142.7 | 871.4 |

**Per-class breakdown**

| Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Negative | 836.1 | 201.1 | 190.9 | 1918.9 | 0.8061 | 0.8141 | 0.8101 |
| Neutral | 802.9 | 317.0 | 284.1 | 1743.0 | 0.7170 | 0.7387 | 0.7277 |
| Positive | 871.4 | 118.6 | 161.6 | 1995.4 | 0.8802 | 0.8436 | 0.8615 |

**Overall metrics** (mean ± sd)

| Metric | Mean | SD |
|---|---|---|
| accuracy | 0.7977 | 0.0062 |
| macro_averaged_accuracy | 0.8651 | 0.0041 |
| macro_precision | 0.8033 | 0.0054 |
| macro_recall | 0.7988 | 0.0064 |
| macro_f1 | 0.7995 | 0.0060 |
| weighted_precision | 0.8019 | 0.0051 |
| weighted_recall | 0.7977 | 0.0062 |
| weighted_f1 | 0.7982 | 0.0061 |

## proposed-projection-morph (30 runs)

**Mean confusion matrix** (rows = true label, columns = predicted)

| True \\ Predicted | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 841.5 | 169.2 | 16.2 |
| Neutral | 191.1 | 797.2 | 98.7 |
| Positive | 21.2 | 146.4 | 865.4 |

**Per-class breakdown**

| Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Negative | 841.5 | 212.3 | 185.5 | 1907.7 | 0.7985 | 0.8194 | 0.8088 |
| Neutral | 797.2 | 315.7 | 289.8 | 1744.3 | 0.7163 | 0.7334 | 0.7248 |
| Positive | 865.4 | 114.9 | 167.6 | 1999.1 | 0.8828 | 0.8377 | 0.8597 |

**Overall metrics** (mean ± sd)

| Metric | Mean | SD |
|---|---|---|
| accuracy | 0.7957 | 0.0072 |
| macro_averaged_accuracy | 0.8638 | 0.0048 |
| macro_precision | 0.8012 | 0.0063 |
| macro_recall | 0.7968 | 0.0072 |
| macro_f1 | 0.7975 | 0.0072 |
| weighted_precision | 0.7997 | 0.0061 |
| weighted_recall | 0.7957 | 0.0072 |
| weighted_f1 | 0.7962 | 0.0073 |

## Statistical comparisons

**proposed-projection vs baseline** (PRIMARY, Ch3-F.3, Ha: calamanCy model outperforms baseline, n = 30 matched seeds)

| Metric | Mean diff | 95% CI | Wins | t | two-sided p | one-tailed p |
|---|---|---|---|---|---|---|
| macro F1 | +0.0016 | [-0.0008, +0.0039] | 20/30 | 1.369 | 0.1815 | 0.0908 |
| weighted F1 | +0.0016 | [-0.0008, +0.0039] | 20/30 | 1.352 | 0.1867 | 0.0934 |

**proposed-projection-morph vs baseline** (secondary -- no hypothesis-mandated direction, n = 30 matched seeds)

| Metric | Mean diff | 95% CI | Wins | t | two-sided p | one-tailed p |
|---|---|---|---|---|---|---|
| macro F1 | -0.0004 | [-0.0033, +0.0025] | 15/30 | -0.267 | 0.7911 | 0.6044 |
| weighted F1 | -0.0004 | [-0.0033, +0.0025] | 15/30 | -0.283 | 0.7794 | 0.6103 |

**proposed-projection vs proposed-projection-morph** (secondary -- no hypothesis-mandated direction, n = 30 matched seeds)

| Metric | Mean diff | 95% CI | Wins | t | two-sided p | one-tailed p |
|---|---|---|---|---|---|---|
| macro F1 | +0.0020 | [-0.0005, +0.0044] | 21/30 | 1.663 | 0.1070 | 0.0535 |
| weighted F1 | +0.0020 | [-0.0004, +0.0044] | 20/30 | 1.681 | 0.1035 | 0.0517 |

## One-sample t-test vs benchmark (weighted F1 = 0.84)

One-tailed direction tested: mean > 0.84, per SOP3/Objective 3 ("outperforms"). Hypothesis 2 as worded ("comparable to") is not a one-sample-t-test claim; see docs section 10.

| Model | n | Mean weighted F1 | Diff vs benchmark | t | two-sided p | one-tailed p (>0.84) |
|---|---|---|---|---|---|---|
| baseline | 30 | 0.7966 | -0.0434 | -42.277 | 1.308e-27 | 1 |
| proposed-projection | 30 | 0.7982 | -0.0418 | -37.823 | 3.123e-26 | 1 |
| proposed-projection-morph | 30 | 0.7962 | -0.0438 | -33.010 | 1.483e-24 | 1 |
