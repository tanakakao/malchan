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


def test_single_output_pipeline_class_is_defined_in_pipeline_module():
    sys.modules.pop("malchan.models.models", None)

    from malchan.pipeline.single_output import SingleOutputMLModelPipeline

    assert SingleOutputMLModelPipeline.__module__ == "malchan.pipeline.single_output"
    assert "malchan.models.models" not in sys.modules


def test_pipeline_classes_live_in_pipeline_modules():
    """Verify pipeline classes are defined in their canonical modules."""

    from malchan.pipeline.single_output import SingleOutputMLModelPipeline
    from malchan.pipeline.multi_output import MLModelPipeline

    assert SingleOutputMLModelPipeline.__module__ == "malchan.pipeline.single_output"
    assert MLModelPipeline.__module__ == "malchan.pipeline.multi_output"


def test_pipeline_package_exports_pipeline_classes():
    """Verify the pipeline package exports canonical pipeline classes."""

    from malchan.pipeline import SingleOutputMLModelPipeline, MLModelPipeline

    assert SingleOutputMLModelPipeline.__module__ == "malchan.pipeline.single_output"
    assert MLModelPipeline.__module__ == "malchan.pipeline.multi_output"


def test_models_package_reexports_pipeline_classes():
    """Verify the models package re-exports canonical pipeline classes."""

    from malchan.models import SingleOutputMLModelPipeline, MLModelPipeline

    assert SingleOutputMLModelPipeline.__module__ == "malchan.pipeline.single_output"
    assert MLModelPipeline.__module__ == "malchan.pipeline.multi_output"


def test_models_models_reexports_pipeline_classes():
    """Verify the legacy models module re-exports canonical pipeline classes."""

    from malchan.pipeline.single_output import SingleOutputMLModelPipeline as single_pipeline_cls
    from malchan.pipeline.multi_output import MLModelPipeline as multi_pipeline_cls
    from malchan.models.models import SingleOutputMLModelPipeline as single_legacy_cls
    from malchan.models.models import MLModelPipeline as multi_legacy_cls

    assert single_legacy_cls is single_pipeline_cls
    assert multi_legacy_cls is multi_pipeline_cls


def test_pipeline_modules_do_not_delegate_to_legacy_models_module():
    """Verify pipeline modules do not construct instances through legacy wrappers."""

    from malchan.pipeline.single_output import SingleOutputMLModelPipeline
    from malchan.pipeline.multi_output import MLModelPipeline

    assert SingleOutputMLModelPipeline.__new__ is object.__new__
    assert MLModelPipeline.__new__ is object.__new__
