from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A  # type: ignore

class BirdDataset(Dataset):
    """
    PyTorch Dataset for bird species crops described in CSV rows.

    Each item corresponds to one cropped bird image and its species label.
    CSV rows must contain at least:
      - `crop_path`: path to the crop image (relative to `img_root` or absolute)
      - a label column (configurable, default: `species_name`)
    Additional fields (e.g., `source_image`, `bbox_original`) are preserved in
    `rows` but not used directly here.
    """

    def __init__(
        self,
        rows: List[Dict],
        cls2id: Dict[str, int],
        img_root: Path,
        transform: A.Compose,
        missing_size: int = 224,
        label_key: str = "species_name",
    ) -> None:
        """
        Initialize the dataset.

        Args:
            rows:
                List of record dicts, typically produced from a split CSV
                via `DataFrame.to_dict(orient="records")`. Each row must
                include at least `crop_path` and the label column.
            cls2id:
                Mapping from label name (string) to integer class index.
                This is used to convert the label into a tensor.
            img_root:
                Root directory under which `crop_path` is resolved. If
                `crop_path` is absolute, `img_root` is ignored.
            transform:
                Albumentations `Compose` object to apply to the loaded image
                (e.g., training or evaluation pipeline).
            missing_size:
                Spatial size (pixels) of the fallback black image used when an
                image file cannot be read. Defaults to 224.
            label_key:
                Column name to use for class labels. Default: "species_name".
                For 2025 dataset with remapped labels, use "remapped_label".
        """
        self.rows: List[Dict] = rows
        self.cls2id: Dict[str, int] = cls2id
        self.img_root: Path = img_root
        self.tf: A.Compose = transform
        self.missing_size: int = missing_size
        self.label_key: str = label_key

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample by index.

        Steps:
          1. Look up the metadata row and map `species_name` to a class index.
          2. Resolve `crop_path` to an absolute filesystem path.
          3. Load the image with OpenCV (BGR), convert to RGB.
          4. Apply the Albumentations transform pipeline.
          5. Convert the result to a PyTorch tensor with shape (3, H, W).

        Returns:
            x: Float32 image tensor of shape (3, H, W).
            y: Long tensor containing the class index (scalar).
        """
        row = self.rows[idx]

        # Convert label string into an integer class index.
        y_idx = self.cls2id[row[self.label_key]]

        # Normalize path separators and build a Path object.
        p_raw = row["crop_path"].replace("\\", "/")
        p0 = Path(p_raw)

        # Resolve to an absolute path: use as-is if absolute, otherwise
        # join to the configured image root directory.
        #
        # Note: some metadata rows already store paths like "crops/CLASS/file.jpg",
        # while IMAGE_ROOT is "data/crops". To avoid "data/crops/crops/...",
        # strip a leading "crops" component before joining.
        if p0.is_absolute():
            full_path = p0
        else:
            parts = p0.parts
            if len(parts) > 0 and parts[0].lower() == "crops":
                relative = Path(*parts[1:])
            else:
                relative = p0
            full_path = (self.img_root / relative).resolve()

        # Read the image from disk in BGR format.
        img = cv2.imread(full_path.as_posix())
        if img is None:
            # If the file is missing or unreadable, emit a warning for
            # the first few occurrences and fall back to a black image
            # so that training can continue.
            if idx < 3:
                print(f"[warn] missing image -> {full_path.as_posix()}")
            img = np.zeros((self.missing_size, self.missing_size, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB for consistency with
            # standard computer vision models.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Albumentations pipeline; returns a dict with an "image" key.
        img_aug = self.tf(image=img)["image"]

        # Convert HxWxC numpy array to CxHxW PyTorch tensor.
        x = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
        y = torch.tensor(y_idx, dtype=torch.long)

        return x, y
