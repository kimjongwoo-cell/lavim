"""PVCR errors remain mutable so context managers can attach tracebacks."""


class LayoutError(ValueError):
    """The native cache or visual metadata violates a PVCR contract."""
