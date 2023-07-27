import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from gazebo_msgs.msg import ModelStates
import numpy as np
import math
import visual_utils.open3d_vis_utils as V
import glob
from pathlib import Path




class get_Lidar:
    def __init__(self):
        self.ext='.txt'
        self.suber=rospy.Subscriber('/velodyne_points2',PointCloud2,self.lidar_callbck)
        self.suber_label=rospy.Subscriber('/gazebo/model_states',ModelStates,self.label_callbck)
        self.count=1
        self.count_label=1
        data_file_list = glob.glob(str(Path('/home/sim2real/3DSSD-torch-master/data/airplane_2187/training') / f'*{self.ext}'))
        data_file_list.sort()
        self.train_flie_list=data_file_list
        self.data_save_path='/home/sim2real/3DSSD-torch-master/data/airplane_2187/training/'+str(len(self.train_flie_list)).rjust(4,'0')+self.ext
        self.label_save_path='/home/sim2real/3DSSD-torch-master/data/airplane_2187/training/label/'+str(len(self.train_flie_list)).rjust(4,'0')+self.ext



    def lidar_callbck(self,data):
        if self.count == 1:
            self.count = 2
            # print(data['head'])
            lidar=point_cloud2.read_points_list(data,field_names=("x", "y", "z"))
            lidar=np.asarray(lidar,dtype=np.float32)
            V.draw_scenes(
                points=lidar,ref_boxes=self.box
            )
            print('debug: get_lidar.py 37\n','len points:',len(lidar))
            # V.draw_scenes(
            #     points=lidar
            # )

            np.savetxt(self.data_save_path,lidar,delimiter=',')
            np.savetxt(self.label_save_path,self.box,delimiter=',')
            # print('num_points:',len(lidar))
            print(len(self.train_flie_list),'ok')
            # ok_=input('ok?')
            # self.count=1
            # self.count_label=1





    def label_callbck(self,data):
        if self.count_label == 1:
            self.count_label = 2
        # if self.count == 1:
        #     self.count = 2
            # print(data)
            pos=data.pose[6].position
            orientation=data.pose[6].orientation
            # print(pos,orientation)
            head=math.atan2(2*(orientation.w*orientation.z+orientation.x*orientation.y),1-2*(orientation.y**2+orientation.z**2))
            box=np.asarray([pos.x,pos.y,pos.z,86.2,66.6,21.9,head-1.5707],dtype=np.float32)
            print('box',box)
            self.box=box.reshape(-1,7)

def main():
    rospy.init_node('get_lidar',anonymous=True)
    get_lidar=get_Lidar()
    # print(get_lidar.train_flie_list)
    rospy.spin()
if __name__ =='__main__':
    main()
