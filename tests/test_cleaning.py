"""Uji cleaning (PRD §7.3), imputasi median, dan scaler."""
import numpy as np
import pandas as pd
import pytest

from qos.features.preprocess import clean_features, fit_scaler, impute_median

COLS = ["a", "b"]


def test_clean_features_urutan_dan_median():
    df = pd.DataFrame({"a": [1.0, 1.0, np.inf, -5.0, 3.0, 5.0], "b": [2.0, 2.0, 4.0, 1.0, np.nan, 6.0]})
    out, med = clean_features(df, COLS, max_rows=100, random_state=42)
    # baris 1 duplikat, baris 3 negatif dibuang; inf → NaN lalu diimputasi
    assert len(out) == 4
    assert med == {"a": 3.0, "b": 4.0}  # median a [1,3,5], b [2,4,6]
    assert not out[COLS].isna().any().any()
    assert out["a"].tolist() == [1.0, 3.0, 3.0, 5.0]


def test_clean_features_sampling_reproducible():
    df = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2})
    o1, _ = clean_features(df, COLS, 10, 42)
    o2, _ = clean_features(df, COLS, 10, 42)
    assert len(o1) == 10 and o1.equals(o2)


def test_clean_features_kolom_kosong():
    with pytest.raises(ValueError):
        clean_features(pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]}), COLS, 10, 42)


def test_impute_median_live():
    X = pd.DataFrame({"a": [np.nan, 1.0], "b": [2.0, np.nan]})
    assert impute_median(X, {"a": 9.0, "b": 8.0}).values.tolist() == [[9.0, 2.0], [1.0, 8.0]]


def test_fit_scaler_zscore():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 60.0]})
    Z = fit_scaler(X).transform(X)
    assert Z.mean(axis=0) == pytest.approx([0, 0], abs=1e-12)
    assert Z.std(axis=0) == pytest.approx([1, 1])


def test_build_features_log1p_subset():
    from qos.features.preprocess import build_features

    df = pd.DataFrame({"a": [0.0, np.e - 1], "b": [5.0, 7.0], "c": [1.0, 2.0]})
    out = build_features(df, ["a", "b"], ["a"])
    assert list(out.columns) == ["a", "b"]
    assert out["a"].tolist() == pytest.approx([0.0, 1.0])
    assert out["b"].tolist() == [5.0, 7.0]  # tidak di-log
    assert df["a"].tolist() == [0.0, np.e - 1]  # input tidak berubah


def test_build_features_validasi():
    from qos.features.preprocess import build_features

    df = pd.DataFrame({"a": [-1.0, 2.0], "b": [1.0, 2.0]})
    with pytest.raises(ValueError):
        build_features(df, ["a", "b"], ["a"])  # negatif
    with pytest.raises(ValueError):
        build_features(df, ["a"], ["b"])  # log bukan subset
    with pytest.raises(KeyError):
        build_features(df, ["a", "zzz"], [])
