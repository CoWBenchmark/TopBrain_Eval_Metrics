from results_equal import results_equal


def test_results_equal():
    # Basic equality
    assert results_equal({"a": 1, "b": "x"}, {"a": 1, "b": "x"})
    assert not results_equal({"a": 1}, {"a": 2})
    assert not results_equal({"a": 1}, {"b": 1})
    assert not results_equal({"a": 1, "b": 2}, {"a": 1})
    assert results_equal({}, {})

    # Nested dicts
    assert results_equal({"x": {"y": {"z": 1.0}}}, {"x": {"y": {"z": 1.0}}})
    assert not results_equal({"x": {"y": {"z": 1.0}}}, {"x": {"y": {"z": 2.0}}})
    assert not results_equal({"x": {"y": 1}}, {"x": 1})

    # Float tolerance
    assert results_equal(1.0, 1.0 + 1e-12)
    assert not results_equal(1.0, 1.1)
    assert results_equal({"val": 1.0000000001}, {"val": 1.0})
    assert results_equal({"val": 1}, {"val": 1.0})
    assert results_equal([1.0], [1.0 + 1e-10])
    assert results_equal({"x": 1}, {"x": 1.0000000001})

    # NaN handling
    assert results_equal(float("nan"), float("nan"))
    assert results_equal({"x": float("nan")}, {"x": float("nan")})
    assert results_equal([float("nan")], [float("nan")])
    assert not results_equal(float("nan"), 1.0)
    assert results_equal({"a": float("nan")}, {"a": float("nan")})
    assert results_equal([1.0, float("nan")], [1.0, float("nan")])

    # Lists / tuples
    assert results_equal({"x": [1, 2, 3]}, {"x": [1, 2, 3]})
    assert not results_equal({"x": [1, 2]}, {"x": [1, 3]})
    assert results_equal([1, 2, 3], [1, 2, 3])
    assert results_equal([1.0, 2.0000000001], [1.0, 2.0])
    assert not results_equal([1, 2, 3], [1, 2])
    assert results_equal([1, 2], (1, 2))
    assert results_equal([{"x": 1.0}, {"y": 2.0000000001}], [{"x": 1.0}, {"y": 2.0}])
    assert results_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    assert results_equal([], [])
    assert not results_equal([], [1])

    # Booleans
    assert results_equal(True, True)
    assert not results_equal(True, 1)
    assert not results_equal(False, 0)
    assert results_equal({"flag": True}, {"flag": True})
    assert not results_equal({"flag": True}, {"flag": 1})
    assert not results_equal({"x": True}, {"x": 1})

    # Strings
    assert results_equal("hello", "hello")
    assert not results_equal("hello", "world")
    assert not results_equal("1", 1)
    assert results_equal("café", "café")
    assert not results_equal("café", "cafe")
    assert results_equal("", "")

    # None
    assert results_equal(None, None)
    assert not results_equal(None, 1)
    assert not results_equal(None, 0)
    assert not results_equal(None, "")
    assert not results_equal(None, False)

    # None nested inside a dict
    assert results_equal({"a": None}, {"a": None})
    assert not results_equal({"a": None}, {"a": 1})
    assert not results_equal({"a": 1}, {"a": None})

    # None vs NaN -- these are different concepts, must not match
    assert not results_equal(None, float("nan"))
    assert not results_equal(float("nan"), None)

    # None inside a list
    assert results_equal([1, None, 3], [1, None, 3])
    assert not results_equal([1, None, 3], [1, 2, 3])

    # None mixed with nan in the same structure
    a = {"score": None, "loss": float("nan")}
    b = {"score": None, "loss": float("nan")}
    assert results_equal(a, b)

    b_bad = {"score": 0, "loss": float("nan")}  # None replaced with 0
    assert not results_equal(a, b_bad)

    # mismatched types
    assert not results_equal({"a": 1}, 1)
    a = {"x": 1}
    b = 42
    assert not results_equal(a, b)
    assert not results_equal([1, 2], {"a": 1})

    # Key order doesn't matter
    assert results_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})

    # Self comparison
    a = {"x": {"y": [1, 2, float("nan")]}}
    assert results_equal(a, a)

    # Zero edge cases
    assert results_equal(0.0, -0.0)
    assert results_equal(0.0, 1e-13)
    assert not results_equal(0.0, 1e-6)

    # Large numbers
    assert results_equal(1_000_000.0, 1_000_000.0001)
    assert not results_equal(1_000_000.0, 1_000_100.0)

    # Deep nesting
    assert results_equal(
        {"l1": {"l2": {"l3": {"l4": {"l5": 1.0000000001}}}}},
        {"l1": {"l2": {"l3": {"l4": {"l5": 1.0}}}}},
    )
    assert not results_equal(
        {"l1": {"l2": {"l3": {"l4": {"l5": 1.0}}}}},
        {"l1": {"l2": {"l3": {"l4": {"l5": 2.0}}}}},
    )

    # Mixed types in same dict / realistic example
    a = {
        "model": "v1",
        "metrics": {"accuracy": 0.9999999999, "loss": float("nan")},
        "params": [0.1, 0.2, 0.30000000001],
        "converged": True,
    }
    b = {
        "model": "v1",
        "metrics": {"accuracy": 1.0, "loss": float("nan")},
        "params": [0.1, 0.2, 0.3],
        "converged": True,
    }
    assert results_equal(a, b)
    assert not results_equal({"score": 1.0}, {"score": "1.0"})
    assert not results_equal(
        {"model": "v1", "metrics": {"accuracy": 0.5}},
        {"model": "v1", "metrics": {"accuracy": 0.9}},
    )


def test_results_equal_nested_nan_cases():
    # Deeply nested dict with nan that should match
    a = {
        "run_id": "exp_42",
        "config": {"lr": 0.001, "epochs": 10},
        "results": {
            "fold_1": {"accuracy": 0.87, "loss": float("nan")},
            "fold_2": {"accuracy": 0.9000000001, "loss": 0.234},
        },
    }
    b = {
        "run_id": "exp_42",
        "config": {"lr": 0.001, "epochs": 10},
        "results": {
            "fold_1": {"accuracy": 0.87, "loss": float("nan")},
            "fold_2": {"accuracy": 0.9, "loss": 0.234},
        },
    }
    assert results_equal(a, b)

    # Same structure, but one nan replaced with a real number -> not equal
    b_bad = {
        "run_id": "exp_42",
        "config": {"lr": 0.001, "epochs": 10},
        "results": {
            "fold_1": {"accuracy": 0.87, "loss": 0.0},  # was nan
            "fold_2": {"accuracy": 0.9, "loss": 0.234},
        },
    }
    assert not results_equal(a, b_bad)

    # nan buried inside a list inside nested dicts -> equal
    a2 = {
        "model": "v3",
        "layers": [
            {"name": "l1", "weights": [0.1, 0.2, float("nan")]},
            {"name": "l2", "weights": [float("nan"), 0.5]},
        ],
    }
    b2 = {
        "model": "v3",
        "layers": [
            {"name": "l1", "weights": [0.1, 0.2, float("nan")]},
            {"name": "l2", "weights": [float("nan"), 0.5]},
        ],
    }
    assert results_equal(a2, b2)

    # same but one nan is actually a different number -> not equal
    b2_bad = {
        "model": "v3",
        "layers": [
            {"name": "l1", "weights": [0.1, 0.2, float("nan")]},
            {"name": "l2", "weights": [0.0, 0.5]},  # nan replaced with 0.0
        ],
    }
    assert not results_equal(a2, b2_bad)

    # one dict has nan, other has None in same spot -> not equal (different types)
    a3 = {"metrics": {"score": float("nan")}}
    b3 = {"metrics": {"score": None}}
    assert not results_equal(a3, b3)

    # nan vs missing key entirely -> not equal
    a4 = {"metrics": {"score": float("nan"), "extra": 1}}
    b4 = {"metrics": {"score": float("nan")}}
    assert not results_equal(a4, b4)

    # multiple nans scattered at different depths, all matching -> equal
    a5 = {
        "a": float("nan"),
        "b": {"c": float("nan"), "d": [1.0, float("nan"), 3.0]},
        "e": [{"f": float("nan")}, {"g": 2.0}],
    }
    b5 = {
        "a": float("nan"),
        "b": {"c": float("nan"), "d": [1.0, float("nan"), 3.0]},
        "e": [{"f": float("nan")}, {"g": 2.0000000001}],
    }
    assert results_equal(a5, b5)

    # same shape as above, but one nested numeric value is meaningfully different
    b5_bad = {
        "a": float("nan"),
        "b": {"c": float("nan"), "d": [1.0, float("nan"), 3.0]},
        "e": [{"f": float("nan")}, {"g": 2.5}],  # 2.0 -> 2.5, real difference
    }
    assert not results_equal(a5, b5_bad)

    # nan compared against a very close-to-zero float -> not equal (nan isn't "close" to anything)
    assert not results_equal(float("nan"), 1e-15)

    # two nans at top level inside otherwise-identical large nested structures
    a6 = {
        "experiment": "baseline",
        "seeds": [1, 2, 3],
        "per_seed_results": {
            "1": {"acc": 0.812, "std": float("nan")},
            "2": {"acc": 0.799, "std": 0.0123},
            "3": {"acc": float("nan"), "std": 0.0456},
        },
    }
    a6_copy = {
        "experiment": "baseline",
        "seeds": [1, 2, 3],
        "per_seed_results": {
            "1": {"acc": 0.812, "std": float("nan")},
            "2": {"acc": 0.799, "std": 0.0123},
            "3": {"acc": float("nan"), "std": 0.0456},
        },
    }
    assert results_equal(a6, a6_copy)

    # flip one acc from nan to an actual value -> not equal
    a6_bad = {
        "experiment": "baseline",
        "seeds": [1, 2, 3],
        "per_seed_results": {
            "1": {"acc": 0.812, "std": float("nan")},
            "2": {"acc": 0.799, "std": 0.0123},
            "3": {"acc": 0.75, "std": 0.0456},  # nan -> real value
        },
    }
    assert not results_equal(a6, a6_bad)

    assert results_equal(
        {
            "result": {
                "data": [
                    1.0,
                    {"value": 2.0},
                    {"nan": float("nan")},
                ]
            }
        },
        {
            "result": {
                "data": [
                    1.0 + 1e-10,
                    {"value": 2.0 + 1e-10},
                    {"nan": float("nan")},
                ]
            }
        },
    )

    assert not results_equal(
        {
            "result": {
                "data": [
                    1.0,
                    {"value": 2.0},
                    {"nan": float("nan")},
                ]
            }
        },
        {
            "result": {
                "data": [
                    1.0 + 1e-10,
                    {"value": 3.0},  # genuinely different
                    {"nan": float("nan")},
                ]
            }
        },
    )
