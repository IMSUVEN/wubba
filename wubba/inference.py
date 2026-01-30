from dataclasses import asdict
from typing import List

import torch
import torch.nn.functional as F
from lightning import Trainer
from torch.utils.data import DataLoader

from wubba.config import Config
from wubba.data import HTMLDataProcessor, HTMLLayoutDataset
from wubba.model import WubbaLightningModule


class WubbaInference:
    """A wrapper for running inference with a trained Wubba model.

    This class simplifies the process of loading a model from a checkpoint
    and using it to generate embeddings for new HTML documents.

    Attributes:
        config: The configuration object used for the model.
        model: The loaded WubbaLightningModule.
        trainer: A Lightning Trainer instance for running prediction.
    """

    def __init__(
        self,
        model_path: str,
        config: Config = None,
        use_compile: bool = True,
    ):
        """Initializes the inference wrapper.

        Args:
            model_path: Path to the .ckpt model checkpoint.
            config: A Config object. If None, a default config is used.
            use_compile: Whether to use `torch.compile` for faster inference.
        """
        self.config = config or Config()
        self.model_path = model_path
        self.use_compile = use_compile
        self.model = None
        self.trainer = None

        self.data_processor = HTMLDataProcessor(
            max_depth=self.config.max_depth,
            max_position=self.config.max_position,
            max_sequence_length=self.config.max_sequence_length,
        )

    def _load_model(self):
        """Loads the model from checkpoint and prepares it for inference."""
        if self.model is not None:
            return

        self.model = WubbaLightningModule.load_from_checkpoint(
            self.model_path,
            map_location=self.config.device,
            # Pass kwargs for model initialization
            **asdict(self.config),
        )
        self.model.eval()

        if self.use_compile:
            self.model = torch.compile(self.model)

        self.trainer = Trainer(
            accelerator="auto",
            devices=1,  # Inference on a single device
            precision=self.config.mixed_precision,
            logger=False,
            enable_checkpointing=False,
        )

    @torch.inference_mode()
    def predict(
        self, html_documents: List[str], batch_size: int = 1024
    ) -> torch.Tensor:
        """Generates embeddings for a list of HTML documents.

        Args:
            html_documents: A list of raw HTML strings.
            batch_size: The batch size to use for inference.

        Returns:
            A tensor of shape (num_documents, embedding_dim) containing the
            L2-normalized embeddings.
        """
        self._load_model()  # Lazy loading

        dataset = HTMLLayoutDataset(
            html_documents,
            transform=self.data_processor.html_clean_to_tensor,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        # The predict method returns a list of batches
        predictions = self.trainer.predict(self.model, dataloader)
        if not predictions:
            return torch.empty(0, self.config.transformer_dim)

        # Concatenate batches and normalize
        embeddings = torch.cat(predictions, dim=0)
        return F.normalize(embeddings, p=2, dim=-1)
