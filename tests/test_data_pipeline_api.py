"""Smoke: API de data_pipeline delega en feature_pipeline."""
import pytest

pytest.importorskip("dotenv")

from unittest.mock import MagicMock, patch

import pandas as pd

from src.pipelines.data_pipeline import run_ingestion_to_features_pipeline


def test_run_ingestion_to_features_delegates():
    df = pd.DataFrame({"x": [1]})
    fs = MagicMock()
    fake_out = {"ok": True}

    with patch(
        "src.pipelines.data_pipeline.run_feature_pipeline", return_value=fake_out
    ) as mock_run:
        out = run_ingestion_to_features_pipeline(df, fs, "fs-001_baseline")

    mock_run.assert_called_once_with(df, fs, "fs-001_baseline")
    assert out == fake_out
