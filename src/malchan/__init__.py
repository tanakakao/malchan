"""Top-level public API for :mod:`malchan`.

The project has several optional-heavy submodules.  Keep top-level import light by
resolving public objects lazily only when they are accessed.
"""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "MLModelPipeline": "malchan.models",
    "SingleOutputMLModelPipeline": "malchan.models",
    "feature_names_from_pipeline": "malchan.models.utils",
    "train_and_evaluate_models": "malchan.models.compare",
    "inverse_analysis": "malchan.inverse_analysis",
    "show_importances": "malchan.visualization",
    "yy_plot_ml": "malchan.visualization",
    "show_pd_and_ice": "malchan.visualization",
    "show_pd_2d": "malchan.visualization",
    "show_shap_scatter": "malchan.visualization",
    "show_shap_beeswarm": "malchan.visualization",
    "show_ia_result_with_pd": "malchan.visualization",
    "show_ia_result_with_pd_2d": "malchan.visualization",
    "show_ia_importance": "malchan.visualization",
    "show_ia_result_pareto": "malchan.visualization",
    "make_sheet": "malchan.export",
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    """Resolve public objects lazily.

    Args:
        name: Public attribute name requested from the top-level package.

    Returns:
        The resolved object from the corresponding submodule.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'malchan' has no attribute {name!r}")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
