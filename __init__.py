def __getattr__(name):
    if name in ("NODE_CLASS_MAPPINGS", "DISPLAY_NAME_MAPPINGS"):
        from .nodes import NODE_CLASS_MAPPINGS, DISPLAY_NAME_MAPPINGS
        globals()[name] = locals()[name]
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
