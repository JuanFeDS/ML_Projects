"""Tests de selección de etapas del orquestador."""
from src.utils.orchestration import (
    PIPELINE_STAGES,
    select_pipeline_scripts,
)


def test_pipeline_stages_count():
    assert len(PIPELINE_STAGES) == 4


def test_select_full_pipeline():
    assert len(select_pipeline_scripts()) == 4


def test_select_predict_only():
    s = select_pipeline_scripts(predict_only=True)
    assert len(s) == 1
    assert s[0][0] == "04_predict"


def test_select_from_train():
    s = select_pipeline_scripts(from_train=True)
    assert [x[0] for x in s] == ["03_train", "04_predict"]


def test_select_skip_eda():
    s = select_pipeline_scripts(skip_eda=True)
    assert [x[0] for x in s] == ["02_features", "03_train", "04_predict"]
