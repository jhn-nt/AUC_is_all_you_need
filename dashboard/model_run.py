from scipy.stats import beta, rv_continuous
import plotly.express as px
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import os
import logging
import pyarrow.parquet as pq

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
)
from typing import Callable, List, Dict
import dataclasses

from beta_gen import ModelConfig


# METRICS
class Metric:
    def __init__(
        self,
        fn: Callable[[list[int], list[float]], float],
        fit_threshold: bool = False,
        direction: str = "maximize",
    ):
        self.fn = fn
        self.fit_threshold = fit_threshold
        self.direction = direction
        self.threshold = None
        
    @staticmethod
    def true_positives(y_true, y_pred):
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        return np.sum((y_true_np == 1) & (y_pred_np == 1))

    @staticmethod
    def true_negatives(y_true, y_pred):
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        return sum((y_true_np == 0) & (y_pred_np == 0))

    @staticmethod
    def false_positives(y_true, y_pred):
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        return sum((y_true_np == 0) & (y_pred_np == 1))

    @staticmethod
    def false_negatives(y_true, y_pred):
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        return sum((y_true_np == 1) & (y_pred_np == 0))

    def _score(self, data: pl.DataFrame, thresh: float | None = None) -> float:
        labels = list(data["label"].to_numpy())
        scores = list(data["model_prob"].to_numpy())
        if thresh is not None:
            scores = [1.0 if s >= thresh else 0.0 for s in scores]

        return self.fn(labels, scores)

    def __call__(self, data: pl.DataFrame) -> float:
        test_data = data.filter(pl.col("split") == "test")

        if self.fit_threshold:
            valid_data = data.filter(pl.col("split") == "valid")
            threshs = list(sorted(set(valid_data["model_prob"])))

            scores_by_thresh = [self._score(valid_data, t) for t in threshs]
            if self.direction == "maximize":
                best_thresh = threshs[np.argmax(scores_by_thresh)]
            else:
                best_thresh = threshs[np.argmin(scores_by_thresh)]

            return self._score(test_data, best_thresh)
        else:
            return self._score(test_data)

METRICS = {
    "AUROC": Metric(roc_auc_score),
    "AUPRC": Metric(average_precision_score),
    "Brier Score": Metric(brier_score_loss, direction="minimize"),
    "Accuracy": Metric(accuracy_score, True),
    "F1": Metric(f1_score, True),
    "True Positives": Metric(Metric.true_positives, True),
    "True Negatives": Metric(Metric.true_negatives, True),
    "False Positives": Metric(Metric.false_positives, True),
    "False Negatives": Metric(Metric.false_negatives, True),
}

def call_metrics(
    df: pl.DataFrame,
    metrics: dict[str, Callable[[pl.Series, pl.Series], float]],
    container: dict[str, list[float]],
    **row_params,
):
    for m, fn in metrics.items():
        container["metric"].append(m)
        container["score"].append(fn(df))
        for k, v in row_params.items():
            container[k].append(v)

def get_metrics(
    data: pl.DataFrame, metrics: dict[str, Callable[[pl.Series, pl.Series], float]]
) -> pl.DataFrame:
    metric_results = {"metric": [], "score": [], "subgroup": [], "sample": []}
    all_subgroups = data.select("subgroup").unique().to_numpy().tolist()
    all_samples = data.select("sample").unique().to_numpy().tolist()
    for sample in all_samples:
        df = data.filter(data["sample"] == sample)
        call_metrics(df, metrics, metric_results, subgroup="ALL", sample=sample)

        for subgroup in all_subgroups:
            subgroup_df = df.filter(df["subgroup"] == subgroup)
            call_metrics(
                subgroup_df,
                metrics,
                metric_results,
                subgroup=str(subgroup),
                sample=sample,
            )
    return pl.DataFrame(metric_results)


@dataclasses.dataclass
class ExperimentConfig:
    model_configs: list[ModelConfig]

    N_test: int
    N_validation: int
    subgroup_prevalences: list[float]
    prevalence: list[float]
    N_samples: int = 5

    def run(
        self, metrics: dict[str, Callable[[list[int], list[float]], float]] = METRICS
    ) -> pl.DataFrame:
        all_metrics = []
        for j, model in enumerate(self.model_configs):
            data = model.get_samples(
                N_test=self.N_test,
                N_validation=self.N_validation,
                subgroup_prevalences=self.subgroup_prevalences,
                prevalence=self.prevalence,
                N_samples=self.N_samples,
            )
            all_metrics.append(
                get_metrics(data, metrics).with_columns(pl.lit(j).alias("model"))
            )

        data = pl.concat(all_metrics)
        data = data.pivot(
            values="score",
            index=["subgroup", "sample", "model"],
            columns="metric",
            aggregate_function=None,
        )
        return data



def run_experiments(model_configs):
    experiment_config = ExperimentConfig(
        model_configs=model_configs,
        N_test=1000,
        N_validation=500,
        subgroup_prevalences=[0.6, 0.4],
        prevalence=[0.7, 0.3],
        N_samples=5,
    )
    results = experiment_config.run()
    return results

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_model_configs() -> List[ModelConfig]:
    """
    Generate a list of model configurations based on specific criteria.
    """
    # More values can be added to these lists to increase the configuration space
    pos_pred_EV_values = [[0.6, 0.55], [0.7, 0.5]]
    pos_pred_confidence_values = [[2, -2], [3, -1]]
    neg_pred_EV_values = [[0.5, 0.45], [0.4, 0.35]]
    neg_pred_confidence_values = [[-2, 2], [-1, 3]]

    configs = []
    for pos_ev in pos_pred_EV_values:
        for pos_conf in pos_pred_confidence_values:
            for neg_ev in neg_pred_EV_values:
                for neg_conf in neg_pred_confidence_values:
                    config = ModelConfig(
                        pos_pred_EV=pos_ev,
                        pos_pred_confidence=pos_conf,
                        neg_pred_EV=neg_ev,
                        neg_pred_confidence=neg_conf
                    )
                    configs.append(config)
    return configs

def save_results(df: pl.DataFrame, idx: int, directory: str = 'results', filename_prefix: str = 'result') -> None:
    """
    Save the experiment results to a parquet file.
    """
    # Ensuring the results directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = os.path.join(directory, f"{filename_prefix}_model_{idx}.parquet")
    pq.write_table(df.to_arrow(), filename)
    logger.info(f"Results saved to {filename}")

if __name__ == "__main__": 
    model_configs = generate_model_configs()
    results = run_experiments(model_configs)

    # Save results for each configuration
    for idx in range(len(model_configs)):
        try:
            save_results(results.filter(pl.col('model') == idx), idx)
        except Exception as e:
            logger.error(f"Failed to save results for model {idx}. Error: {e}")
