import math


def results_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
    """Recursive equality check for JSON-like nested dicts (strings, floats,
    ints, bools, None, lists). NaN equals NaN, floats compared with tolerance."""

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            results_equal(a[k], b[k], rel_tol, abs_tol) for k in a
        )

    # Lists / tuples — recurse elementwise
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            results_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b)
        )

    # Explicitly reject bool vs non-bool numeric type mismatches
    if isinstance(a, bool) != isinstance(b, bool):
        return False

    # Numeric scalars, excluding bool
    if (
        isinstance(a, (int, float))
        and not isinstance(a, bool)
        and isinstance(b, (int, float))
        and not isinstance(b, bool)
    ):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # Exact match fallback (strings, bools, None, mismatched types)
    return a == b
