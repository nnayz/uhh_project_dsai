"""
MLflow logging utilities for DCASE Few-Shot Bioacoustic.

This module provides a unified logging interface that uses MLflow when available
and falls back to console output when not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Try to import MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Get logger for this module - will be captured by Hydra's job_logging
logger = logging.getLogger(__name__)


class MLFlowLoggerWrapper:
    """
    Unified logging interface for MLflow with console fallback.

    This wrapper provides a consistent API for logging parameters, metrics,
    tags, and artifacts. When MLflow is not available or not initialized,
    it falls back to console output.

    Usage:
        logger = MLFlowLoggerWrapper()
        logger.start_run(experiment_name="my_experiment", run_name="run_1")
        logger.log_param("learning_rate", 0.001)
        logger.log_metric("accuracy", 0.95)
        logger.end_run()
    """

    def __init__(self, use_mlflow: bool = True, tracking_uri: Optional[str] = None):
        """
        Initialize the logger.

        Args:
            use_mlflow: Whether to use MLflow (if available).
            tracking_uri: MLflow tracking URI (optional).
        """
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self._run_started = False
        self._experiment_name = None
        self._run_name = None

        if self.use_mlflow and tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def set_tracking_uri(self, uri: str):
        """Set the MLflow tracking URI."""
        if self.use_mlflow:
            mlflow.set_tracking_uri(uri)

    def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        nested: bool = False,
    ):
        """
        Start an MLflow run.

        Args:
            run_name: Name for this run.
            experiment_name: Name of the experiment.
            nested: Whether this is a nested run.
        """
        self._experiment_name = experiment_name
        self._run_name = run_name

        if self.use_mlflow:
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name, nested=nested)
            self._run_started = True
            self.info(f"Started MLflow run: {run_name} (experiment: {experiment_name})")
        else:
            self.info(
                f"MLflow not available. Run: {run_name} (experiment: {experiment_name})"
            )

    def end_run(self):
        """End the current MLflow run."""
        if self.use_mlflow and self._run_started:
            mlflow.end_run()
            self._run_started = False
            self.info("Ended MLflow run")

    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        if self.use_mlflow and self._run_started:
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                self.warning(f"Failed to log param {key}: {e}")
        logger.info(f"[PARAM] {key}: {value}")

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        if self.use_mlflow and self._run_started:
            try:
                # MLflow has a limit on param value length, so convert to string
                safe_params = {k: str(v)[:250] for k, v in params.items()}
                mlflow.log_params(safe_params)
            except Exception as e:
                self.warning(f"Failed to log params: {e}")
        for key, value in params.items():
            logger.info(f"[PARAM] {key}: {value}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if self.use_mlflow and self._run_started:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                self.warning(f"Failed to log metric {key}: {e}")

        if step is not None:
            logger.info(f"[METRIC] {key}: {value} (step={step})")
        else:
            logger.info(f"[METRIC] {key}: {value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        if self.use_mlflow and self._run_started:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                self.warning(f"Failed to log metrics: {e}")

        for key, value in metrics.items():
            if step is not None:
                logger.info(f"[METRIC] {key}: {value} (step={step})")
            else:
                logger.info(f"[METRIC] {key}: {value}")

    def set_tag(self, key: str, value: str):
        """Set a single tag."""
        if self.use_mlflow and self._run_started:
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                self.warning(f"Failed to set tag {key}: {e}")
        logger.info(f"[TAG] {key}: {value}")

    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags."""
        if self.use_mlflow and self._run_started:
            try:
                mlflow.set_tags(tags)
            except Exception as e:
                self.warning(f"Failed to set tags: {e}")
        for key, value in tags.items():
            logger.info(f"[TAG] {key}: {value}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if self.use_mlflow and self._run_started:
            try:
                if Path(local_path).exists():
                    mlflow.log_artifact(local_path, artifact_path)
                    logger.info(f"[ARTIFACT] Logged: {local_path}")
                else:
                    self.warning(f"Artifact not found: {local_path}")
            except Exception as e:
                self.warning(f"Failed to log artifact {local_path}: {e}")
        else:
            logger.info(f"[ARTIFACT] {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts."""
        if self.use_mlflow and self._run_started:
            try:
                if Path(local_dir).exists():
                    mlflow.log_artifacts(local_dir, artifact_path)
                    logger.info(f"[ARTIFACTS] Logged directory: {local_dir}")
                else:
                    self.warning(f"Artifacts directory not found: {local_dir}")
            except Exception as e:
                self.warning(f"Failed to log artifacts from {local_dir}: {e}")
        else:
            logger.info(f"[ARTIFACTS] {local_dir}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log a dictionary as a JSON artifact."""
        if self.use_mlflow and self._run_started:
            try:
                mlflow.log_dict(dictionary, artifact_file)
                logger.info(f"[DICT] Logged to: {artifact_file}")
            except Exception as e:
                self.warning(f"Failed to log dict to {artifact_file}: {e}")
        else:
            logger.info(f"[DICT] {artifact_file}: {dictionary}")

    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure as an artifact."""
        if self.use_mlflow and self._run_started:
            try:
                mlflow.log_figure(figure, artifact_file)
                logger.info(f"[FIGURE] Logged: {artifact_file}")
            except Exception as e:
                self.warning(f"Failed to log figure to {artifact_file}: {e}")
        else:
            logger.info(f"[FIGURE] {artifact_file}")

    def info(self, message: str):
        """Log an info message."""
        logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        logger.error(message)

    def debug(self, message: str):
        """Log a debug message."""
        logger.debug(message)

    @property
    def is_mlflow_available(self) -> bool:
        """Check if MLflow is available."""
        return MLFLOW_AVAILABLE

    @property
    def is_run_active(self) -> bool:
        """Check if an MLflow run is active."""
        return self._run_started

    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.use_mlflow and self._run_started:
            return mlflow.active_run().info.run_id
        return None

    def get_experiment_id(self) -> Optional[str]:
        """Get the current experiment ID."""
        if self.use_mlflow and self._run_started:
            return mlflow.active_run().info.experiment_id
        return None


# Create a global logger instance
_global_logger: Optional[MLFlowLoggerWrapper] = None


def get_logger(
    use_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
) -> MLFlowLoggerWrapper:
    """
    Get or create the global MLflow logger.

    Args:
        use_mlflow: Whether to use MLflow.
        tracking_uri: MLflow tracking URI.

    Returns:
        MLFlowLoggerWrapper instance.
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = MLFlowLoggerWrapper(
            use_mlflow=use_mlflow,
            tracking_uri=tracking_uri,
        )
    return _global_logger


def reset_logger():
    """Reset the global logger."""
    global _global_logger
    if _global_logger is not None and _global_logger.is_run_active:
        _global_logger.end_run()
    _global_logger = None
