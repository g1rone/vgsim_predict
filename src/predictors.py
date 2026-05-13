"""Simple predictors for migration intensities."""

from __future__ import annotations

import numpy as np


def predict_equal_mu(populations_number: int, mu: float, include_diagonal: bool = False) -> np.ndarray:
    """Return a migration matrix with equal migration intensity values.

    Parameters
    ----------
    populations_number : int
        Number of populations/demes.
    mu : float
        Common migration intensity/probability value.
    include_diagonal : bool, default=False
        If False, diagonal values are set to zero.

    Returns
    -------
    np.ndarray
        Matrix of shape (populations_number, populations_number).
    """
    if populations_number <= 0:
        raise ValueError("populations_number must be positive")
    if mu < 0:
        raise ValueError("mu must be non-negative")

    prediction = np.full((populations_number, populations_number), float(mu), dtype=float)

    if not include_diagonal:
        np.fill_diagonal(prediction, 0.0)

    return prediction


def predict_equal_mu_from_bounds(
    populations_number: int,
    lower_bound: float,
    upper_bound: float,
    include_diagonal: bool = False,
) -> np.ndarray:
    """Return equal-mu prediction using the midpoint of the interval."""
    if lower_bound < 0 or upper_bound < 0:
        raise ValueError("bounds must be non-negative")
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must be less than or equal to upper_bound")

    mu = (float(lower_bound) + float(upper_bound)) / 2.0
    return predict_equal_mu(populations_number, mu, include_diagonal=include_diagonal)