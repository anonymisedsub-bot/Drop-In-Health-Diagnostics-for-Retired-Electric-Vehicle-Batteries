"""The four supported group-3 methods and their stable result identities."""

# method name, campaign key, result-directory alias
_METHODS = (
    ("DA-AL", "da_al", "SSL (Adversarial)_mlp"),
    ("DA-MMD", "da_mmd", "SSL (MMD)_mlp"),
    ("DA-CORAL", "da_coral", "SSL (CORAL)_mlp"),
    ("SSL-PL", "da_pl", "SSL (PL)_mlp"),
)
GROUP3_METHODS = tuple(method for method, _, _ in _METHODS)
GROUP3_COMBOS = tuple((method, "mlp") for method in GROUP3_METHODS)
GROUP3_CAMPAIGN_KEYS = tuple(key for _, key, _ in _METHODS)
GROUP3_RESULT_ALIASES = {method: alias for method, _, alias in _METHODS}


def select_group3_methods(method_names=None) -> tuple[str, ...]:
    """Validate a subset of the fixed four methods; None selects all four."""
    if method_names is None:
        return GROUP3_METHODS
    selected = (method_names,) if isinstance(method_names, str) else tuple(method_names)
    if not selected:
        raise ValueError("Select at least one group-3 method")
    unsupported = [name for name in selected if name not in GROUP3_METHODS]
    if unsupported:
        raise ValueError(f"Unsupported group-3 methods: {unsupported}; choose from {GROUP3_METHODS}")
    if len(set(selected)) != len(selected):
        raise ValueError("Group-3 method selections must not contain duplicates")
    return selected
