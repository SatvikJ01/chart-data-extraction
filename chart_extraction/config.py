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
class PipelineConfig:
    image_dir: Path = Path("/kaggle/input/benetech-making-graphs-accessible/test/images")
    image_glob: str = "*.jpg"

    donut_model_dir: Path = Path("/kaggle/input/benetech-donut")
    x_axis_model_path: Path = Path("/kaggle/input/x-axis-model-10/model (1).pth")
    y_axis_model_path: Path = Path("/kaggle/input/y-axis-model-10/Y_Point_generation_weights_1.0.pth")
    marker_model_path: Path = Path("/kaggle/input/marker-model/Marker_weights.pth")

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
