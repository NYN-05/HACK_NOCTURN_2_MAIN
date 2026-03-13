from __future__ import annotations

import warnings


def suppress_noisy_warnings() -> None:
    """Suppress common non-actionable warnings during training/inference runs."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\.|$)")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision(\.|$)")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib(\.|$)")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"PIL(\.|$)")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"threadpoolctl(\.|$)")

    # Optional dependency warning type from scikit-learn; keep this local to avoid hard dependency.
    try:
        from sklearn.exceptions import UndefinedMetricWarning

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    except Exception:
        pass
