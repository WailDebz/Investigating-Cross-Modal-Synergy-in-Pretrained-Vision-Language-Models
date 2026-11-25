from .contracts import (
    FusionHead,
    ModalityEncoder,
    ModelBatch,
    ParameterSummary,
    adapt_batch,
    inference,
    seeded_model_init,
    summarize_parameters,
)
from .fusion import (
    BilinearFusionHead,
    EmbeddingClassifier,
    LateFusionHead,
    LinearFusionHead,
    MLPFusionHead,
)
from .foundation import DEFAULT_CLIP_MODEL, FrozenCLIP
from .adaptation import AdaptiveCLIPClassifier, configure_final_block_adaptation
from .scratch import (
    ScratchClassifier,
    ScratchImageEncoder,
    ScratchTextEncoder,
    TextVocabulary,
    scratch_input_dims,
)

__all__ = [
    "FusionHead",
    "ModalityEncoder",
    "ModelBatch",
    "ParameterSummary",
    "adapt_batch",
    "inference",
    "seeded_model_init",
    "summarize_parameters",
    "BilinearFusionHead",
    "EmbeddingClassifier",
    "LateFusionHead",
    "LinearFusionHead",
    "MLPFusionHead",
    "ScratchClassifier",
    "ScratchImageEncoder",
    "ScratchTextEncoder",
    "TextVocabulary",
    "scratch_input_dims",
    "DEFAULT_CLIP_MODEL",
    "FrozenCLIP",
    "AdaptiveCLIPClassifier",
    "configure_final_block_adaptation",
]
