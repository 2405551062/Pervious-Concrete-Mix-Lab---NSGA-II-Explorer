import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from models import evaluate_mix_from_vars


class PerviousMixProblem(ElementwiseProblem):
    def __init__(self, aggregate_size_mm=14.0, glass_ratio_norm=0.5, cha_ratio_norm=0.5):
        # two variables: S (total substitution 0-1), wbr
        xl = np.array([0.0, 0.05])
        xu = np.array([1.0, 0.07])
        super().__init__(n_var=2, n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.aggregate_size_mm = aggregate_size_mm
        self.glass_ratio_norm = glass_ratio_norm
        self.cha_ratio_norm = cha_ratio_norm

    def _evaluate(self, x, out, *args, **kwargs):
        r = evaluate_mix_from_vars(x,
                                   aggregate_size_mm=self.aggregate_size_mm,
                                   glass_ratio_norm=self.glass_ratio_norm,
                                   cha_ratio_norm=self.cha_ratio_norm)
        # we minimize by convention
        f1 = -r["strength_mpa"]
        f2 = -r["permeability_mms"]
        f3 = r["co2_kg_per_m3"]
        out["F"] = [f1, f2, f3]


def run_nsga2(pop_size=80, n_gen=60, seed=None, aggregate_size_mm=14.0, glass_ratio_norm=0.5, cha_ratio_norm=0.5):
    problem = PerviousMixProblem(aggregate_size_mm=aggregate_size_mm,
                                 glass_ratio_norm=glass_ratio_norm,
                                 cha_ratio_norm=cha_ratio_norm)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=False,
        verbose=False,
    )

    # collect Pareto solutions
    X = res.X
    F = res.F
    details = [evaluate_mix_from_vars(x,
                                     aggregate_size_mm=aggregate_size_mm,
                                     glass_ratio_norm=glass_ratio_norm,
                                     cha_ratio_norm=cha_ratio_norm) for x in X]
    return {"X": X, "F": F, "details": details}
