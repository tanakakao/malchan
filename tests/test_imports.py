def test_import_malchan_is_lightweight():
    import malchan

    assert malchan.__version__ == "0.1.0"
    assert "MLModelPipeline" in malchan.__all__
    assert "ModelComparisonResult" in malchan.__all__
    assert "MultiOutputModelComparisonResult" in malchan.__all__
    assert "make_sheet" in malchan.__all__
