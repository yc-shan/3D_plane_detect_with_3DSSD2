

import numpy as np

from pathlib import Path

from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.dataset import DatasetTemplate
import glob
import argparse
from pcdet.config import cfg, cfg_from_yaml_file


class airplaneDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path='/home/sim2real/3DSSD-torch-master/data/airplane_2187/training', logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        root_path=Path(root_path)
        self.root_path = root_path
        self.ext = '.txt'


        data_file_list = glob.glob(str(root_path / f'*{self.ext}'))

        data_file_list.sort()
        self.sample_file_list = data_file_list
        label_file_list=glob.glob(str(root_path /'label'/ f'*{self.ext}'))
        label_file_list.sort()
        self.label_file_list=label_file_list



    def get_lidar(self, idx):

        assert Path(self.sample_file_list[idx]).exists()
        lidar=np.loadtxt(self.sample_file_list[idx],dtype=np.float32,delimiter=',')
        lidar=np.pad(lidar,((0,0),(0,1)))
        return lidar



    def get_label(self, idx):

        assert Path(self.label_file_list[idx]).exists()
        label=np.loadtxt(self.label_file_list[idx],dtype=np.float32,delimiter=',')
        return label.reshape(-1,7)


    def __len__(self):

        return len(self.sample_file_list)

    def __getitem__(self, index):
        points=self.get_lidar(index)

        input_dict = {
            'points': points,
            'frame_id': index,
            'gt_names': np.asarray(['airplane']),

        }
        try:
            box=self.get_label(index)
            input_dict.update({
                'gt_boxes': box
            })
        except:
            pass

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/sim2real/3DSSD-torch-master/output/3dssd/airplane/3dssd.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/sim2real/3DSSD-torch-master/data/airplane_2187/training',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


if __name__ == '__main__':
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = airplaneDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path),  logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    print(demo_dataset[0])
