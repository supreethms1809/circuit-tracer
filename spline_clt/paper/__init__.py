"""Config-driven paper evaluation utilities for Spline CLT."""

from spline_clt.paper.config import SuiteConfig, load_suite_config
from spline_clt.paper.runner import PaperSuiteRunner

__all__ = ["PaperSuiteRunner", "SuiteConfig", "load_suite_config"]
