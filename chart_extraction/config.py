"""Central configuration.

Replaces the two divergent ``CFG`` classes in the original notebooks
(``tuned-donut.ipynb`` and ``inference-3.ipynb``), which disagreed on model
directory and generation settings while sharing a class name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GenerationConfig:
    """Donut decoding settings.

    AUDIT NOTE (Phase 0, finding E)
    -------------------------------
    The original notebooks set ``temperature``/``top_k``/``top_p`` but never set
    ``do_sample``, which defaults to ``False`` in ``transformers``. Those three
    parameters are therefore **inert** in both notebooks: sampling parameters are
    only consulted when ``do_sample=True``. The only operative difference between
    the two notebook configs was ``num_beams`` (1 vs 2).

    This class makes ``do_sample`` explicit so the distinction is visible rather
    than implied. Sampling parameters are only emitted into the generate() kwargs
    when sampling is actually enabled -- see :meth:`to_generate_kwargs`.
    """

    max_length: int = 512
    num_beams: int = 1
    do_sample: bool = False
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    early_stopping: bool = True

    def to_generate_kwargs(self) -> dict:
        """Build kwargs for ``model.generate``.

        Sampling parameters are omitted entirely when ``do_sample`` is False, so
        that an inert setting can never be silently carried along and later
        mistaken for a tuned one.
        """
        kwargs: dict = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
        }
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        return kwargs


# The two generation configs the notebooks actually used, preserved so Phase 1
# can score them against each other. Note that they differ ONLY in num_beams.
GREEDY = GenerationConfig(num_beams=1, do_sample=False)
BEAM2 = GenerationConfig(num_beams=2, do_sample=False)


@dataclass(frozen=True)
class RuntimeConfig:
    """Device, precision and OOM behaviour.

    Separate from PipelineConfig because these are properties of the *machine*
    a run happens on, not of the pipeline being evaluated. Two runs with
    different RuntimeConfigs should produce the same scores; two runs with
    different PipelineConfigs should not.

    The exception, and it is recorded in the result file for exactly this
    reason: an OOM retry that falls back to a lower Donut input resolution DOES
    change that image's prediction. That is why OomPolicy events are counted and
    reported rather than merely logged.
    """

    device: str = "cuda:0"
    #: "fp32" or "fp16". fp16 halves Donut's weights and autocasts the
    #: detection stages.
    precision: str = "fp32"
    #: Retry ladder for images that will not fit. Empty disables rescaling.
    oom_retry_scales: tuple[float, ...] = (0.75, 0.5)
    oom_retry_enabled: bool = True
    #: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at startup.
    expandable_segments: bool = True

    @classmethod
    def local_gpu(cls, device: str = "cuda:0") -> "RuntimeConfig":
        """Preset for a single local card: fp16 and full OOM recovery."""
        return cls(device=device, precision="fp16")

    @classmethod
    def cpu(cls) -> "RuntimeConfig":
        return cls(device="cpu", precision="fp32", expandable_segments=False)


#: Batch sizes for a single local GPU. Batch size 1 everywhere: peak memory on
#: this pipeline is dominated by Donut's encoder activations, and a batch of one
#: is what lets a small card finish a run at all. It is slower per image, which
#: is why the latency figures record the batch size they were measured at.
LOCAL_GPU_BATCH_SIZES = {
    "donut_batch_size": 1,
    "axis_batch_size": 1,
    "marker_batch_size": 1,
}


@dataclass(frozen=True)
class PipelineConfig:
    # Paths default to None, never to a Kaggle mount. A hardcoded
    # /kaggle/input default silently resolves to a non-existent path on any
    # other machine and fails later inside a model loader, which is a much
    # worse error than being told up front that nothing was configured.
    # Resolution lives in chart_extraction.paths (CLI > env > config > preset).
    image_dir: Path | None = None
    image_glob: str = "*.jpg"

    donut_model_dir: Path | None = None
    x_axis_model_path: Path | None = None
    y_axis_model_path: Path | None = None
    marker_model_path: Path | None = None

    generation: GenerationConfig = field(default_factory=lambda: GREEDY)

    donut_batch_size: int = 4
    axis_batch_size: int = 32
    marker_batch_size: int = 4
    num_workers: int = 2

    # Side length that both the axis CNN and the marker detector resize to.
    # All downstream pixel geometry lives in this coordinate space.
    working_resolution: int = 256

    axis_max_num_points: int = 25
    marker_num_classes: int = 4
    marker_label_id: int = 3
    marker_score_threshold: float = 0.5

    #: "full" (Donut + axis CNN + marker detector) or "donut_only" (Donut's
    #: generated series used directly, no detection stages). See
    #: chart_extraction.stages. A donut_only run is a different system
    #: producing a different number, not a degraded full run, so the mode is
    #: recorded alongside every score.
    mode: str = "full"

    # --- The AxisLabelSource seam (Phase 0 finding B) -----------------------
    # Selects which implementation supplies y-axis tick *labels* to the
    # calibration stage. See chart_extraction.axis.labels for the registry and
    # for why the default is knowingly incorrect.
    axis_label_source: str = "donut_series"

    # Inference-time determinism. The notebooks passed random_padding=True, a
    # train-time augmentation, making predictions non-reproducible.
    donut_random_padding: bool = False

    placeholder_data_series: str = "0;0"
    placeholder_chart_type: str = "line"

    def require_paths(self, *names: str) -> None:
        """Raise if any named path field is unset.

        Called by anything that is about to load a checkpoint, so an unset path
        is reported by name rather than surfacing as a confusing failure inside
        transformers or torch.load.
        """
        unset = [name for name in names if getattr(self, name) is None]
        if unset:
            raise ValueError(
                f"PipelineConfig paths not set: {unset}. Resolve them with "
                "chart_extraction.paths.resolve_paths (CLI flag, BENETECH_* env "
                "var, config file, or a preset)."
            )
