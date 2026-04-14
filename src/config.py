from dataclasses import dataclass

# Column names — single source of truth for the dataset schema
COL_IMAGE = "image_name"
COL_LABEL = "instance_label"
COL_X1 = "bbox_x_tl"
COL_Y1 = "bbox_y_tl"
COL_X2 = "bbox_x_br"
COL_Y2 = "bbox_y_br"
COL_SOURCE = "source"

COORD_COLS = [COL_X1, COL_Y1, COL_X2, COL_Y2]
REQUIRED_COLUMNS = {COL_IMAGE, COL_LABEL, COL_X1, COL_Y1, COL_X2, COL_Y2}
OUTPUT_COLS = [COL_IMAGE, COL_LABEL, COL_X1, COL_Y1, COL_X2, COL_Y2]
QUESTIONABLE_SUFFIX = "_questionable"


@dataclass
class MergeConfig:
    iou_threshold: float = 0.5
    dist_threshold: float = 20.0
    no_questionable: bool = False
    output: str = "merged.csv"
    compute_kappa: bool = False
    image_width: int | None = None
    image_height: int | None = None

    def validate(self) -> None:
        if not (0.0 < self.iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be in (0, 1], got {self.iou_threshold}")
        if self.dist_threshold < 0:
            raise ValueError(f"dist_threshold must be >= 0, got {self.dist_threshold}")
        if self.image_width is not None and self.image_width <= 0:
            raise ValueError(f"image_width must be > 0, got {self.image_width}")
        if self.image_height is not None and self.image_height <= 0:
            raise ValueError(f"image_height must be > 0, got {self.image_height}")
