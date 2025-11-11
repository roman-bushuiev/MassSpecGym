import typing as T
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional, Callable
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from massspecgym.data.datasets import MassSpecDataset


class MassSpecDataModule(pl.LightningDataModule):
    """
    Data module containing a mass spectrometry dataset. This class is responsible for loading, splitting, and wrapping
    the dataset into data loaders according to pre-defined train, validation, test folds.
    """

    def __init__(
        self,
        dataset: MassSpecDataset,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = True,
        split_pth: Optional[Path] = None,
        worker_init_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        **kwargs
    ):
        """
        Args:
            split_pth (Optional[Path], optional): Path to a .tsv file with columns "identifier" and "fold",
                corresponding to dataset item IDs, and "fold", containg "train", "val", "test"
                values. Default is None, in which case the split from the `dataset` is used.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.worker_init_fn = worker_init_fn
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Pre-processing to be executed only on a single main device when using distributed training."""
        pass

    def _prepare_split_mask(self):

        # Prepare split
        if self.split_pth is None:
            df_split = self.dataset.metadata[["identifier", "fold"]]
        else:
            # NOTE: custom split is not tested
            df_split = pd.read_csv(self.split_pth, sep="\t")
            if set(df_split.columns) != {"identifier", "fold"}:
                raise ValueError('Split file must contain "id" and "fold" columns.')
            df_split["identifier"] = df_split["identifier"].astype(str)
            if set(self.dataset.metadata["identifier"]) != set(df_split["identifier"]):
                raise ValueError(
                    "Dataset item IDs must match the IDs in the split file."
                )

        df_split = df_split.set_index("identifier")["fold"]
        if not set(df_split) <= {"train", "val", "test"}:
            raise ValueError(
                '"Folds" column must contain only "train", "val", or "test" values.'
            )

        # Split dataset
        self.split_mask = df_split.loc[self.dataset.metadata["identifier"]].values

    def _split_dataset(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Subset(
                self.dataset, np.where(self.split_mask == "train")[0]
            )
            self.val_dataset = Subset(self.dataset, np.where(self.split_mask == "val")[0])
        elif stage == "test":
            self.test_dataset = Subset(self.dataset, np.where(self.split_mask == "test")[0])
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def setup(self, stage=None):
        """Pre-processing to be executed on every device when using distributed training."""
        self._prepare_split_mask()
        self._split_dataset(stage)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )
