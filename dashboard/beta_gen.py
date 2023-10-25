from scipy.stats import beta, rv_continuous
import plotly.express as px
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import pyarrow

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
)
from typing import Callable

import dataclasses


def beta_from_params(EV: float, conf: float) -> rv_continuous:
    conf = 2**conf
    a = np.sqrt(EV * conf / (1 - EV))
    b = conf / a
    return beta(a, b)


@dataclasses.dataclass
class ModelConfig:
    pos_pred_EV: list[float]
    pos_pred_confidence: list[float]
    neg_pred_EV: list[float]
    neg_pred_confidence: list[float]

    def __post_init__(self):
        for n in [
            "pos_pred_EV",
            "pos_pred_confidence",
            "neg_pred_EV",
            "neg_pred_confidence",
        ]:
            val = getattr(self, n)
            if type(val) is not list:
                raise TypeError(f"{n} must be a list. Found {type(val)}")
            if len(val) != self.n_subgroups:
                raise ValueError(
                    f"All distribution parameters must have the same number of subgroups ({self.n_subgroups}). "
                    f"{n} has {len(val)}."
                )
            for v in val:
                if n.endswith("_EV") and (v <= 0 or v >= 1):
                    raise ValueError(
                        f"{n} must be a list of floats v such that 0 < v < 1. Found {v}"
                    )

    @property
    def n_subgroups(self) -> int:
        return len(self.pos_pred_EV)

    @property
    def pos_rvs(self):
        return [
            beta_from_params(ev, conf)
            for ev, conf in zip(self.pos_pred_EV, self.pos_pred_confidence)
        ]

    @property
    def neg_rvs(self):
        return [
            beta_from_params(ev, conf)
            for ev, conf in zip(self.neg_pred_EV, self.neg_pred_confidence)
        ]

    def subgroup_viz(self, N_plot_samples: int = 100):
        plot_data = {"subgroup": [], "X": [], "Y": [], "prob_type": [], "label": []}
        for subgroup in range(self.n_subgroups):
            pos_rv = self.pos_rvs[subgroup]
            neg_rv = self.neg_rvs[subgroup]

            X = np.arange(0.0, 1.0, 1 / N_plot_samples)
            pos_pdf = pos_rv.pdf(X)
            neg_pdf = neg_rv.pdf(X)
            likelihood_of_pos = pos_pdf / (pos_pdf + neg_pdf)

            for x, p, n, L in zip(X, pos_pdf, neg_pdf, likelihood_of_pos):
                plot_data["subgroup"].extend([subgroup for _ in range(3)])
                plot_data["X"].extend([x for _ in range(3)])
                plot_data["Y"].extend([p, n, L])
                plot_data["prob_type"].extend(
                    ["p(f(x) | Y = 1)", "p(f(x) | Y = 0)", "P(Y = 1 | f(x) = p)"]
                )
                plot_data["label"].extend([f"Subgroup {subgroup + 1}" for _ in range(3)])

        return px.line(
            plot_data, 
            x="X", 
            y="Y", 
            facet_col="prob_type", 
            color="label",
            title="Subgroup Visualizations",
            labels={"label": "Subgroups", "X": "Probability", "Y": "Density", "prob_type": "Probability Type"},
            template="plotly_dark"
        )



    def get_samples(
        self,
        N_test: int,
        N_validation: int,
        subgroup_prevalences: list[float],
        prevalence: list[float],
        N_samples: int = 5,
    ) -> pl.DataFrame:
        assert sum(subgroup_prevalences) == 1
        assert len(subgroup_prevalences) == self.n_subgroups

        data = {
            "subgroup": [],
            "label": [],
            "model_prob": [],
            "sample": [],
            "split": [],
        }

        N_test_per_subgroup = []
        N_valid_per_subgroup = []

        for members_frac in subgroup_prevalences[:-1]:
            N_test_per_subgroup.append(int(round(members_frac * N_test)))
            N_valid_per_subgroup.append(int(round(members_frac * N_validation)))

        N_test_per_subgroup.append(N_test - sum(N_test_per_subgroup))
        N_valid_per_subgroup.append(N_validation - sum(N_valid_per_subgroup))

        for i, N_test_subgroup, N_valid_subgroup, prev, pos_rv, neg_rv in zip(
            range(self.n_subgroups),
            N_test_per_subgroup,
            N_valid_per_subgroup,
            prevalence,
            self.pos_rvs,
            self.neg_rvs,
        ):
            N_pos_subgroup_test = int(round(prev * N_test_subgroup))
            N_neg_subgroup_test = N_test_subgroup - N_pos_subgroup_test
            N_pos_subgroup_valid = int(round(prev * N_valid_subgroup))
            N_neg_subgroup_valid = N_valid_subgroup - N_pos_subgroup_valid

            pos_probs = pos_rv.rvs(
                (N_pos_subgroup_test + N_pos_subgroup_valid) * N_samples
            )
            neg_probs = neg_rv.rvs(
                (N_neg_subgroup_test + N_neg_subgroup_valid) * N_samples
            )

            subgroup_IDs = [i for _ in range(N_test_subgroup + N_valid_subgroup)]
            labels = [1 for _ in range(N_pos_subgroup_test + N_pos_subgroup_valid)] + [
                0 for _ in range(N_neg_subgroup_test + N_neg_subgroup_valid)
            ]
            splits = (
                ["test" for _ in range(N_pos_subgroup_test)]
                + ["valid" for _ in range(N_pos_subgroup_valid)]
                + ["test" for _ in range(N_neg_subgroup_test)]
                + ["valid" for _ in range(N_neg_subgroup_valid)]
            )

            pos_list = list(pos_probs)
            neg_list = list(neg_probs)

            data["subgroup"].extend(subgroup_IDs * N_samples)
            data["label"].extend(labels * N_samples)
            data["split"].extend(splits * N_samples)
            for j in range(N_samples):
                data["sample"].extend([j for _ in subgroup_IDs])
                data["model_prob"].extend(pos_list[j : len(pos_list) : N_samples])
                data["model_prob"].extend(neg_list[j : len(neg_list) : N_samples])

        return pl.DataFrame(data)

