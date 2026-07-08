def test_import_pipeline_package_is_lightweight():
    import malchan.pipeline as pipeline

    assert "SingleOutputMLModelPipeline" in pipeline.__all__
    assert "MLModelPipeline" in pipeline.__all__


def test_top_level_pipeline_exports_are_declared():
    import malchan

    assert "SingleOutputMLModelPipeline" in malchan.__all__
    assert "MLModelPipeline" in malchan.__all__
