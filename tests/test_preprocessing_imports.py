def test_import_preprocessing_package_is_lightweight():
    import malchan.preprocessing as preprocessing

    assert "make_pipeline" in preprocessing.__all__
    assert "make_preprocess" in preprocessing.__all__
