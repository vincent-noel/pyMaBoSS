"""
    Fitting the parameters of a MaBoSS model by minimizing a user-defined scoring function.

    This module only uses the public Simulation API (copy, update_parameters, run),
    so it can also be imported as a standalone file.
"""

import math
import time

import pandas as pd
import scipy.optimize


class _ToleranceReached(Exception):
    pass


class FitResult:
    """
        Result of :func:`fit_parameters`

        :ivar parameters: dict of fitted parameter values (rounded)
        :ivar score: score of the fitted parameters
        :ivar success: True if the score reached the requested tolerance
        :ivar model: copy of the model with the fitted parameters
        :ivar result: the simulation result of the fitted parameters
        :ivar history: DataFrame of all the simulated parameter sets and their scores
        :ivar optimizer_result: the scipy OptimizeResult (None if stopped on tolerance)
    """

    def __init__(self, parameters, score, success, model, result, history, optimizer_result):
        self.parameters = parameters
        self.score = score
        self.success = success
        self.model = model
        self.result = result
        self.history = history
        self.optimizer_result = optimizer_result

    def __repr__(self):
        return "FitResult(score=%g, success=%s, n_simulations=%d,\n  parameters=%s)" % (
            self.score, self.success, len(self.history), self.parameters
        )


def round_significant(value, digits):
    """
        Rounds a value to a number of significant digits

        >>> round_significant(0.01234, 1)
        0.01
        >>> round_significant(1234, 2)
        1200.0
    """
    if value == 0 or digits is None:
        return float(value)
    return float("%.*e" % (digits - 1, value))


def fit_parameters(model, score, parameters, bounds=(1e-2, 1e2), significant_digits=1, tol=1e-2,
                   method="differential_evolution", log_scale=True, verbose=True, **optimizer_kwargs):
    """
        Fits model parameters by minimizing a scoring function

        :param model: the model (Simulation) to fit. It is not modified.
        :param score: function taking the simulation result and returning the score to minimize
        :param parameters: list of parameters to perturb (ex: ["$u_CyclinB", "$d_CyclinB"])
        :param bounds: (low, high) for all parameters, or dict {parameter: (low, high)}
        :param significant_digits: parameter values are rounded to this number of
            significant digits before simulating (None: no rounding)
        :param tol: the fit stops as soon as the score is below this value
        :param method: "differential_evolution" (default), "dual_annealing", "direct", "shgo",
            or any scipy.optimize.minimize method (ex: "Nelder-Mead", "Powell")
        :param log_scale: optimize the log10 of the parameters (default True)
        :param verbose: print each improvement
        :param optimizer_kwargs: passed to the scipy optimizer (ex: maxiter, popsize, seed)
        :return: a :class:`FitResult`

        The simulation seed is kept constant, so that the score only changes with the
        parameters. Use thread_count in the model's parameters to parallelize simulations,
        and restrict the model's outputs to the nodes the score needs.

        Example : fitting the probability of a node at the last time point

        >>> def score(result):
        ...     return abs(result.get_last_nodes_probtraj()["Proliferation"].iloc[0] - 0.3)
        >>> res = fit_parameters(model, score, ["$u_CyclinB", "$d_CyclinB"], bounds=(0.1, 10), tol=0.01)
    """

    parameters = list(parameters)
    for p in parameters:
        if p not in model.param:
            raise ValueError("Unknown parameter %s" % p)

    if isinstance(bounds, dict):
        bounds_list = [tuple(bounds[p]) for p in parameters]
    else:
        bounds_list = [tuple(bounds)] * len(parameters)
    if log_scale:
        if any(low <= 0 for low, _ in bounds_list):
            raise ValueError("Bounds must be strictly positive with log_scale=True")
        opt_bounds = [(math.log10(low), math.log10(high)) for low, high in bounds_list]
    else:
        opt_bounds = bounds_list

    def to_values(x):
        values = [10 ** xi for xi in x] if log_scale else list(x)
        return tuple(
            round_significant(min(max(v, low), high), significant_digits)
            for v, (low, high) in zip(values, bounds_list)
        )

    cache = {}
    history = []
    best = {"score": math.inf}
    start = time.time()

    def objective(x):
        values = to_values(x)
        if values not in cache:
            sim = model.copy()
            sim.update_parameters(**dict(zip(parameters, values)))
            result = sim.run()
            value = float(score(result))
            cache[values] = value
            history.append(dict(zip(parameters, values), score=value))

            if value < best["score"]:
                best.update(score=value, values=values, result=result)
                if verbose:
                    print("[%d sims, %.0fs] score=%.4g  %s" % (
                        len(history), time.time() - start, value,
                        ", ".join("%s=%g" % (p, v) for p, v in zip(parameters, values))
                    ), flush=True)
            if value <= tol:
                raise _ToleranceReached()
        return cache[values]

    x0 = []
    for p, (low, high) in zip(parameters, bounds_list):
        v = min(max(float(model.param[p]), low), high)
        x0.append(math.log10(v) if log_scale else v)

    optimizer_result = None
    try:
        if method == "differential_evolution":
            optimizer_kwargs.setdefault("polish", False)  # gradient polishing is useless on rounded values
            optimizer_kwargs.setdefault("x0", x0)
            optimizer_result = scipy.optimize.differential_evolution(objective, opt_bounds, **optimizer_kwargs)
        elif method == "dual_annealing":
            optimizer_kwargs.setdefault("x0", x0)
            optimizer_result = scipy.optimize.dual_annealing(objective, opt_bounds, **optimizer_kwargs)
        elif method in ("direct", "shgo"):
            optimizer_result = getattr(scipy.optimize, method)(objective, opt_bounds, **optimizer_kwargs)
        else:
            optimizer_result = scipy.optimize.minimize(objective, x0, method=method, bounds=opt_bounds, **optimizer_kwargs)
    except _ToleranceReached:
        pass

    fitted = model.copy()
    fitted.update_parameters(**dict(zip(parameters, best["values"])))

    return FitResult(
        parameters=dict(zip(parameters, best["values"])),
        score=best["score"],
        success=best["score"] <= tol,
        model=fitted,
        result=best["result"],
        history=pd.DataFrame(history),
        optimizer_result=optimizer_result,
    )
