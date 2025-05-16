import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats


def cholesky_from_svd(a: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a matrix using SVD and QR.

    This function works with positive semi-definite matrices.

    Parameters
    ----------
    a : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        The Cholesky decomposition of the input matrix.
    """
    u, s, _ = np.linalg.svd(a)
    b = np.diag(np.sqrt(s)) @ u.T
    _, r = np.linalg.qr(b)
    return r.T


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Hermite quadrature points and weights.

    Integration is with respect to the Gaussian density. It corresponds to the
    probabilist's Hermite polynomials.

    Parameters
    ----------
    n: int
        Number of quadrature points.

    Returns
    -------
    knots: array-like
        Gauss-Hermite knots.
    weight: array-like
        Gauss-Hermite weights.
    """
    knots, weights = np.polynomial.hermite.hermgauss(n)
    knots *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    return knots, weights


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


def mean_indicator_pce(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the mean vector `E[coef_indicator_pce_i (a X + b)]` for `i=0,...,n`, and
    where `a, b` two vectors.

    Parameters
    ----------
    n : int
        The order of the PCE.
    a : np.ndarray
        Vector of standard deviations.
    b : np.ndarray
        Vector of means.

    Returns
    -------
    np.ndarray, shape is (n+1, shape of a)
        The mean vector for the PCE.

    Raises
    ------
    ValueError
        If a and b do not have the same shape.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same shape.")

    mean_pce = np.zeros((n + 1, a.shape[0]), dtype=np.float64)
    x = b / (1.0 + a**2) ** 0.5
    for i in range(n + 1):
        if i == 0:
            mean_pce[0, :] = stats.norm.cdf(-x)
        elif i == 1:
            mean_pce[1, :] = (
                np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi) / np.sqrt(1.0 + a**2)
            )
        elif i == 2:
            mean_pce[2, :] = (b / 2.0) * mean_pce[1, :] / (1.0 + a**2)
        else:
            mean_pce[i, :] = (b / i) * mean_pce[i - 1, :] - (
                (i - 2) / (i * (i - 1.0))
            ) * mean_pce[i - 2, :]
            mean_pce[i, :] /= 1.0 + a**2

    return mean_pce


def _sigma0_matrix(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the first row of the PCE covariance matrix
    for a given order n and input vectors a and b.

    Parameters
    ----------
    n : int
        The order of the PCE.
    a : np.ndarray
        Vector of standard deviations.
    b : np.ndarray
        Vector of means.

    Returns
    -------
    np.ndarray
        The sigma0 matrix of size (n+1, K).
    """

    size_a = np.shape(a)[0]
    sigma0_n = np.zeros((n + 1, size_a), dtype=np.float64)
    mu_n = mean_indicator_pce(n=n, a=a, b=b)
    mu_n2 = mean_indicator_pce(n=n, a=a / np.sqrt(1.0 + a**2), b=b / (1.0 + a**2))

    tmp = np.zeros_like(a)
    for k in range(size_a):
        mean_k = [0, 0]
        cov_k = [[1 + a[k] ** 2, a[k] ** 2], [a[k] ** 2, 1 + a[k] ** 2]]
        tmp[k] = stats.multivariate_normal.cdf(x=[-b[k], -b[k]], mean=mean_k, cov=cov_k)  # type: ignore

    sigma0_n[0] = tmp - mu_n[0] ** 2

    if n == 0:
        return sigma0_n

    sigma0_n[1] = mu_n[1] * (mu_n2[0] - mu_n[0])
    if n == 1:
        return sigma0_n

    for i in range(2, n + 1):
        tmp = b * sigma0_n[i - 1, :] - ((i - 2) / (i - 1)) * sigma0_n[i - 2, :]
        tmp -= (a**2) * mu_n[1] * mu_n2[i - 1]
        sigma0_n[i] = tmp / (i * (1 + a**2))

    return sigma0_n


def cov_indicator_pce(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix of a polynomial chaos expansion (PCE).

    The covariance matrix is computed as:
    `Cov[coef_indicator_pce_i (a X + b), coef_indicator_pce_j (a X + b)]`
    for `i,j = 0,...,n`, and where `a` and `b` are two vectors.

    Parameters
    ----------
    n : int
        Order of the PCE.
    a : np.ndarray
        Vector of standard deviations.
    b : np.ndarray
        Vector of means.

    Returns
    -------
    np.ndarray, shape is (n+1, n+1, shape of a)
        Covariance matrix.

    Raises
    ------
    ValueError
        If a and b do not have the same shape.
    """

    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the shape.")

    mean_vector = mean_indicator_pce(n=n, a=a, b=b)

    cov_matrix = np.zeros((n + 1, n + 1, a.shape[0]))
    cov_matrix[0, :, :] = _sigma0_matrix(n=n, a=a, b=b)
    cov_matrix[:, 0, :] = cov_matrix[0, :, :]

    for i in range(0, n):
        for j in range(0, n - 1):
            cov_matrix[i + 1, j + 1, :] = (1 / ((i + 1) * (a**2))) * (
                -(1 + a**2) * (j + 2) * cov_matrix[i, j + 2, :]
                + b * cov_matrix[i, j + 1, :]
                - (j / (j + 1)) * cov_matrix[i, j, :]
            ) - mean_vector[i + 1] * mean_vector[j + 1]

    return cov_matrix


def mean_cov_pce_quad(
    a: np.ndarray, b: np.ndarray, n_pce: int, n_quad: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean vector and covariance matrix of a polynomial chaos expansion (PCE)
    using a Gauss-Hermite quadrature.

    Parameters
    ----------
    a : numpy.ndarray
        Coefficients.
    b : numpy.ndarray
        Coefficients.
    n_pce : int
        Order of the PCE.
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    mean_pce_quad : numpy.ndarray
        Mean vector of the PCE.
    cov_pce_quad : numpy.ndarray
        Covariance matrix of the PCE.
    """

    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    def compute_mean_pce(w_he, y_he_ab, sig):
        """
        Compute the mean vector for the PCE using Gauss-Hermite quadrature.
        """
        mean_pce_quad = np.zeros((n_pce + 1, a.shape[0]))
        for m in range(n_pce + 1):
            if m == 0:
                mean_pce_quad[m] = stats.norm.cdf(-b / (1.0 + a**2) ** 0.5)
            elif m == 1:
                mean_pce_quad[m] = (
                    stats.norm.pdf(b / (1.0 + a**2) ** 0.5) / (1.0 + a**2) ** 0.5
                )
            else:
                mean_pce_quad[m] = (
                    np.sum(
                        w_he[:, None] * special.hermitenorm(m - 1)(y_he_ab),
                        axis=0,
                    )
                    * sig
                    * np.exp(-0.5 * b**2 / (a**2 + 1.0))
                    / (np.sqrt(2.0 * np.pi) * math.factorial(m))
                )
        return mean_pce_quad

    def compute_cov_pce(x_he_ab, y_he_ab, w_he, sig, mean_pce_quad):
        """
        Compute the covariance matrix for the PCE using Gauss-Hermite
        quadrature.
        """
        cov_pce_quad = np.zeros((n_pce + 1, n_pce + 1, a.shape[0]))
        for m1 in range(n_pce + 1):
            if m1 == 0:
                integrand_m1 = coef_indicator_pce(n=0, x=x_he_ab)
            else:
                integrand_m1 = (
                    special.hermitenorm(m1 - 1)(y_he_ab)
                    * sig
                    * np.exp(-0.5 * b**2 / (a**2 + 1.0))
                    / (np.sqrt(2.0 * np.pi) * math.factorial(m1))
                )
            for m2 in range(m1, n_pce + 1):
                if m2 == 0:
                    integrand_m2 = coef_indicator_pce(n=0, x=x_he_ab)
                else:
                    integrand_m2 = (
                        special.hermitenorm(m2 - 1)(y_he_ab)
                        * sig
                        * np.exp(-0.5 * b**2 / (a**2 + 1.0))
                        / (np.sqrt(2.0 * np.pi) * math.factorial(m2))
                    )
                cov_pce_quad[m1, m2, :] = np.sum(
                    w_he[:, None] * integrand_m1 * integrand_m2, axis=0
                )
                cov_pce_quad[m1, m2, :] -= mean_pce_quad[m1, :] * mean_pce_quad[m2, :]

        # Fill in the lower triangular part of the covariance matrix
        cov_pce_quad = np.triu(cov_pce_quad) + np.tril(
            cov_pce_quad.transpose(1, 0, 2), -1
        )
        return cov_pce_quad

    x_he, w_he = gauss_hermite(n=n_quad)
    x_he_ab = a[None, :] * x_he[:, None] + b[None, :]
    mu = -a * b / (1.0 + a**2)
    sig = 1.0 / np.sqrt(1.0 + a**2)
    y_he_ab = a[None, :] * (sig[None, :] * x_he[:, None] + mu[None, :]) + b[None, :]

    # Compute the mean vector and covariance matrix
    mean_pce_quad = compute_mean_pce(w_he, y_he_ab, sig)
    cov_pce_quad = compute_cov_pce(x_he_ab, y_he_ab, w_he, sig, mean_pce_quad)

    return (mean_pce_quad, cov_pce_quad)
