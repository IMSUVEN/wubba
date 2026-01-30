"""Wubba: Self-supervised HTML document embeddings.

Learn layout-invariant representations from raw HTML using contrastive learning,
hierarchical RoPE, and Matryoshka embeddings.
"""

from wubba.config import Config
from wubba.data import HTMLDataProcessor, WubbaDataModule
from wubba.inference import WubbaInference, export_to_onnx, quantize_model
from wubba.metrics import (
    AlignmentUniformityLoss,
    CollapseDetector,
    CollapseStatus,
    EMAModel,
    EmbeddingMetrics,
    SelfPacedWeighter,
)
from wubba.model import (
    EnhancedHybridLoss,
    HybridContrastiveLoss,
    InfoNCELoss,
    MaskedNodePrediction,
    MatryoshkaHybridLoss,
    MatryoshkaProjectionHead,
    SpectralContrastiveLoss,
    StructurePrediction,
    VICRegLoss,
    Wubba,
    WubbaLightningModule,
)
from wubba.train import (
    CollapseMonitorCallback,
    CurriculumLearningCallback,
    EMACheckpointCallback,
    ProgressiveMatryoshkaCallback,
    train,
    train_quick,
)

__all__ = [
    "AlignmentUniformityLoss",
    "CollapseDetector",
    "CollapseMonitorCallback",
    "CollapseStatus",
    "Config",
    "CurriculumLearningCallback",
    "EMACheckpointCallback",
    "EMAModel",
    "EmbeddingMetrics",
    "EnhancedHybridLoss",
    "HTMLDataProcessor",
    "HybridContrastiveLoss",
    "InfoNCELoss",
    "MaskedNodePrediction",
    "MatryoshkaHybridLoss",
    "MatryoshkaProjectionHead",
    "ProgressiveMatryoshkaCallback",
    "SelfPacedWeighter",
    "SpectralContrastiveLoss",
    "StructurePrediction",
    "VICRegLoss",
    "Wubba",
    "WubbaDataModule",
    "WubbaInference",
    "WubbaLightningModule",
    "export_to_onnx",
    "quantize_model",
    "train",
    "train_quick",
]
__version__ = "0.2.0"
