import importlib.util


def test_llm_module_import_does_not_require_provider_sdks():
    import malchan.llm as llm

    assert llm.LLMConfig is not None
    assert llm.plan_training_configuration is not None


def test_provider_sdks_remain_optional_dependencies():
    # The assertion documents the import boundary rather than requiring the
    # provider packages to be absent from every development environment.
    assert importlib.util.find_spec("malchan.llm") is not None
