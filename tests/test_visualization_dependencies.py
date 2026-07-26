from importlib.metadata import version


def test_shap_dependency_stack_imports_on_supported_python() -> None:
    """SHAPのNumba/llvmlite依存が同時にimportできることを確認する。"""
    import llvmlite
    import numba
    import shap

    assert llvmlite is not None
    assert numba is not None
    assert shap is not None
    assert version("llvmlite").startswith("0.48.")
    assert version("numba").startswith("0.66.")
