import argparse
import glob
from pathlib import Path
from sensor_msgs.msg import PointCloud2
import rospy
import numpy as np
import torch
from sensor_msgs import point_cloud2
from pcdet.config import cfg, cfg_from_yaml_file
import time
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import open3d_vis_utils as V
from pcdet.datasets.airplane.airplane_dataset import airplaneDataset
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from sensor_msgs.msg import PointField
from std_msgs.msg import Float32MultiArray
# from . import Gotopose #导航用



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/sim2real/3DSSD-torch-master/output/3dssd/airplane/3dssd.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/sim2real/3DSSD-torch-master/data/airplane/training',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/home/sim2real/3DSSD-torch-master/output/3dssd/airplane/16384/checkpoint_epoch_120.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.txt', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def points_to_msg(points):
    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = "livox_frame"


    msg.height = 1
    msg.width = len(points)

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points, np.float32).tobytes()
    return msg

def main():
    rospy.init_node('pred_lidar', anonymous=True)
    puber=rospy.Publisher('points_in_box',PointCloud2,queue_size=10)
    puber_box=rospy.Publisher('pred_box',Float32MultiArray,queue_size=10)
    # nav_move_to=Gotopose.GoToPose() #导航用

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------start-------------------------')
    demo_dataset = airplaneDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    while not rospy.is_shutdown():
        msg=rospy.wait_for_message('/velodyne_points2',PointCloud2)
        points=point_cloud2.read_points_list(msg,field_names=("x", "y", "z","intensity"))
        points = np.asarray(points, dtype=np.float32)

        # np.savetxt('/home/sim2real/3DSSD-torch-master/data/airplane/training/0250.txt', points, delimiter=',')
        data_dict={
            'points': points,
            'frame_id': 0
        }
        data_dict=demo_dataset.prepare_data(data_dict)
        with torch.no_grad():
            # for idx, data_dict in enumerate(demo_dataset):
            #     data_dict=demo_dataset[-1]
                # logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            t1=time.time()
            pred_dicts, _ = model.forward(data_dict)
            t2=time.time()
            logger.info(f'pred time:{t2-t1}')

            points_cuda=data_dict['points'][:, 1:4]
            points_=np.asarray(points_cuda.cpu())
            box_cuda= pred_dicts[0]['pred_boxes']
            box_=np.asarray(box_cuda.cpu())
            score_=pred_dicts[0]['pred_scores']
            score_=score_.cpu()
            mask=np.argmax(score_)
            box_=box_[mask]
            box_=box_.reshape(-1,7)
            # print(box_cuda)
        t3=time.time()
        logger.info(f'process time:{t3-t2}')
        V.draw_scenes(
            points=points_, ref_boxes=box_,
        )
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            points_cuda.unsqueeze(dim=0), box_cuda[:, 0:7].contiguous().unsqueeze(dim=0)
        ).long().squeeze(dim=0).cpu()
        point_mask=np.asarray(box_idxs_of_pts>=0)
        # print(point_mask)
        # print(points_.shape)
        points_in_box=points_[point_mask][:]
        # print(points_in_box)
        msg=points_to_msg(points_in_box)
        boxmsg=Float32MultiArray(data=box_.reshape(7))
        puber_box.publish(boxmsg)
        puber.publish(msg)


        V.draw_scenes(
            points=points_in_box
        )








if __name__ == '__main__':
    main()
