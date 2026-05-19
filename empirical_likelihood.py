import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import chi2

def empirical_likelihood_ratio(x, mu):
    """
    Compute -2 log R(mu) for a given sample x and hypothesised mean mu.
    Returns test statistic ~ χ²(1) under null.
    """
    n = len(x)
    # Solve for λ such that sum_i (x_i - mu) / (1 + λ (x_i - mu)) = 0
    # The objective function: g(λ) = sum (x_i - mu) / (1 + λ (x_i - mu))
    # We also need λ in interval where all 1+λ(x_i-mu) > 0.
    def objective(l):
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = 1 + l * (x - mu)
            if np.any(denom <= 0):
                return 1e10
            return np.sum((x - mu) / denom)
    # Find root
    try:
        res = root_scalar(objective, bracket=[-1e5, 1e5], method='bisect')
        lmbda = res.root
    except:
        # If no root, return large value (reject null)
        return 1e10
    # Compute -2 log R
    with np.errstate(divide='ignore', invalid='ignore'):
        logR = -2 * np.sum(np.log(1 + lmbda * (x - mu)))
    return max(0.0, logR)   # ensure non‑negative

def el_confidence_interval(x, confidence=0.90):
    """
    Compute empirical likelihood confidence interval for the mean.
    Returns (lower, upper, mean_est).
    """
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    # Use the sample mean as starting point
    x_bar = np.mean(x)
    # Critical value from χ²(1)
    crit = chi2.ppf(confidence, 1)
    # Find lower bound: mu < x_bar such that -2 log R(mu) = crit
    # Find upper bound similarly
    # We'll use a simple grid search around the sample mean
    low = x_bar
    step = np.std(x) / 10
    # Expand downwards
    while low - step > np.min(x) and low > np.min(x):
        test = low - step
        if test < np.min(x):
            break
        lr = empirical_likelihood_ratio(x, test)
        if lr > crit:
            break
        low = test
    # Now refine with root finder
    def lower_obj(mu):
        return empirical_likelihood_ratio(x, mu) - crit
    try:
        if lower_obj(low) * lower_obj(x_bar) < 0:
            res = root_scalar(lower_obj, bracket=[low, x_bar], method='bisect')
            lower = res.root
        else:
            lower = low
    except:
        lower = low
    # Upper bound
    high = x_bar
    while high + step < np.max(x) and high < np.max(x):
        test = high + step
        if test > np.max(x):
            break
        lr = empirical_likelihood_ratio(x, test)
        if lr > crit:
            break
        high = test
    try:
        def upper_obj(mu):
            return empirical_likelihood_ratio(x, mu) - crit
        if upper_obj(x_bar) * upper_obj(high) < 0:
            res = root_scalar(upper_obj, bracket=[x_bar, high], method='bisect')
            upper = res.root
        else:
            upper = high
    except:
        upper = high
    return lower, upper, x_bar

def el_score(x, confidence=0.90):
    """
    Return a score for ranking: the lower bound of the confidence interval.
    This is a conservative estimate of the mean.
    """
    lower, upper, mean_est = el_confidence_interval(x, confidence)
    if np.isnan(lower):
        return 0.0
    return lower
