from .kitti.dataset import KittiDataModule
from .mapillary.dataset import MapillaryDataModule
from .bim.dataset import BimDataModule

modules = {"mapillary": MapillaryDataModule, "kitti": KittiDataModule, "bim":BimDataModule}
