import pandas as pd
from lingam.lewis import LEWIS


def test_get_scores_success():
    df = pd.DataFrame(
        {
            "x1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "x2": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "x3": [0, 3, 0, 3, 3, 3, 0, 3, 2, 1],
            "x4": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "y": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    explainer = LEWIS()
    explainer.get_scores(df, ["x3"], [0], [2], "y")
    explainer.get_scores(df, ["x3"], [3], [0], "y")
    explainer.get_scores(df, ["x3"], [3], [0], "y", [], [], ["x1"])
    explainer.get_scores(df, ["x3"], [3], [0], "y", [], [], ["x1", "x2"])
    explainer.get_scores(df, ["x3"], [3], [0], "y", ["x2"], [1], ["x1"])
    explainer.get_scores(df, ["x1"], [0], [1], "x4")
    explainer.get_scores(df, ["x1"], [0], [1], "x5")


def test_get_score_invalid_inputs():
    df = pd.DataFrame({"X": [0, 1, 0], "Y": [1, 0, 1]})
    explainer = LEWIS()

    # df must be a pandas DataFrame
    try:
        explainer.get_scores(
            df.values,
            x_names=["X"],
            x_values=[1],
            x_prime_values=[0],
            o_name="Y",
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # x_names must be a list of strings
    try:
        explainer.get_scores(
            df,
            x_names=1,
            x_values=[1],
            x_prime_values=[0],
            o_name="Y",
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # x_values must be a list
    try:
        explainer.get_scores(
            df,
            x_names=["X"],
            x_values=1,
            x_prime_values=[0],
            o_name="Y",
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # x_prime_values must be a list
    try:
        explainer.get_scores(
            df,
            x_names=["X"],
            x_values=[1],
            x_prime_values=1,
            o_name="Y",
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # o_name must be a string
    try:
        explainer.get_scores(
            df,
            x_names=["X"],
            x_values=[1],
            x_prime_values=[0],
            o_name=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # k_names must be a list of strings
    try:
        explainer.get_scores(
            df,
            x_names=["X"],
            x_values=[1],
            x_prime_values=[0],
            o_name="Y",
            k_names=1,
            k_values=[],
            c_names=[],
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # k_values must be a list
    try:
        explainer.get_scores(
            df,
            x_names=["X"],
            x_values=[1],
            x_prime_values=[0],
            o_name="Y",
            k_names=[],
            k_values=1,
            c_names=[],
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # c_names must be a list of strings
    try:
        explainer.get_scores(
            df,
            x_names=["X"],
            x_values=[1],
            x_prime_values=[0],
            o_name="Y",
            k_names=[],
            k_values=[],
            c_names=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # x_names and x_values must have the same length
    try:
        explainer.get_scores(
            df,
            x_names=["X", "Z"],
            x_values=[1],
            x_prime_values=[0, 1],
            o_name="Y",
            k_names=[],
            k_values=[],
            c_names=[],
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # x_names and x_prime_values must have the same length
    try:
        explainer.get_scores(
            df,
            x_names=["X", "Z"],
            x_values=[1, 2],
            x_prime_values=[0],
            o_name="Y",
            k_names=[],
            k_values=[],
            c_names=[],
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # k_names and k_values must have the same length
    try:
        explainer.get_scores(
            df,
            x_names=["X", "Z"],
            x_values=[1, 2],
            x_prime_values=[0, 1],
            o_name="Y",
            k_names=["K1"],
            k_values=[],
            c_names=[],
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # c_names must be a list of strings
    try:
        explainer.get_scores(
            df,
            x_names=["X", "Z"],
            x_values=[1, 2],
            x_prime_values=[0, 1],
            o_name="Y",
            k_names=[],
            k_values=[],
            c_names=["C1", 1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError
