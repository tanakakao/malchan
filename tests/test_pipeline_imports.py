import sys


def test_import_pipeline_package_is_lightweight():
    import malchan.pipeline as pipeline

    assert "SingleOutputMLModelPipeline" in pipeline.__all__
    assert "MLModelPipeline" in pipeline.__all__


def test_top_level_pipeline_exports_are_declared():
    import malchan

    assert "SingleOutputMLModelPipeline" in malchan.__all__
    assert "MLModelPipeline" in malchan.__all__


def test_import_models_package_does_not_load_legacy_models_module():
    sys.modules.pop("malchan.models", None)
    sys.modules.pop("malchan.models.models", None)

    import malchan.models as models

    assert "SingleOutputMLModelPipeline" in models.__all__
    assert "MLModelPipeline" in models.__all__
    assert "malchan.models.models" not in sys.modules
