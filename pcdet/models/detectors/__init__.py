from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .ssd3d import SSD3D

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SSD3D': SSD3D,
}


def build_detector(model_cfg, num_class, dataset):
    # print(__all__)
    model = __all__[model_cfg.NAME]( #model_cfg.NAME就是yaml里的MODEL字典的NAME键的值
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
