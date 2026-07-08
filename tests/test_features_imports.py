def test_import_features_package_is_lightweight():
    import malchan.features as features

    assert "chemistry" in features.__all__
    assert "materials" in features.__all__


def test_import_feature_subpackages_is_lightweight():
    import malchan.features.chemistry as chemistry
    import malchan.features.materials as materials

    assert chemistry.__all__ == []
    assert materials.__all__ == []
