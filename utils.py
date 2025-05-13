import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats


def coef_indicator_pce(n: int, x: ArrayLike) -> np.ndarray:
    """
    Compute the polynomial chaos expansion (PCE) coefficient of the random variable
    `1_{x < Z} with Z = N(0,1)`. The PCE coefficient is computed as
    `E[ He_n(Z) 1_{x < Z} ] / n!` where `He_n` is
    the nth order probabilist's Hermite polynomial. It is equal to:
        `normal_cdf(-x)`, if n=0,
        `normal_pdf(x) * He_{n-1}(x) / n!`, otherwise.

    Parameters
    ----------
    n : int
        The order of the PCE.
    x : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The nth order PCE coefficient at x.
    """
    x = np.asarray(x)
    if n == 0:
        return stats.norm.cdf(-x)
    return stats.norm.pdf(x) * special.hermitenorm(n - 1)(x) / math.factorial(n)


def indicator_pce(x, z, n_pce: int) -> np.ndarray:
    """
    Compute the polynomial chaos expansion (PCE) of the indicator function
    `1_{x < Z} with Z = N(0,1)`. The PCE is computed as
    `sum_{n=0}^{N} coef_indicator_pce(n, x) * He_n(Z)` where `He_n` is
    the nth order probabilist's Hermite polynomial.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    n_pce : int
        The order of the PCE.

    Returns
    -------
    np.ndarray
        The PCE of the indicator function at x.
    """
    x = np.asarray(x)
    z = np.asarray(z)
    return np.sum(
        [
            coef_indicator_pce(n, x) * special.hermitenorm(n)(z)
            for n in range(n_pce + 1)
        ],
        axis=0,
    )
