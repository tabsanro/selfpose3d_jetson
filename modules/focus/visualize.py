from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    # sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from focus.kafka_producer import KafkaProducer

# constants
LIMBS = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 9],
    [9, 10],
    [10, 11],
    [2, 6],
    [2, 12],
    [6, 7],
    [7, 8],
    [12, 13],
    [13, 14],
]


# Factory Method Pattern
def get_visualizer(config):
    if config.KAFKA.VISUALIZATION.ENABLED:
        return Visualizer(StrategyIsaacSim(config))
    return Visualizer(StrategyPlt3d())


# strategy pattern
class VisualizerStrategy(ABC):
    @abstractmethod
    def visualize(self, obj_info):
        pass

class StrategyPlt3d(VisualizerStrategy):
    def __init__(self):
        pass

    def visualize(self, obj_info):
        with plt.ion():
            fig = plt.figure('3d Pose Estimation', clear=True)
            ax = fig.add_subplot(111, projection='3d')
            
            for obj in obj_info:
                roots_3d = obj['roots_3d'][:3]
                preds_3d = obj['preds_3d'][:, :3]
                age = obj['age']
                gender = obj['gender']
                if age < 20:
                    color = 'lightyellow'   # kid
                elif gender == 0:
                    color = 'pink'          # female
                else:
                    color = 'skyblue'       # male
                self.plot_joints_and_limbs(ax, preds_3d, roots_3d, color)

            # TODO lod 추가하기
            # for obj in obj_info:
            #     lod = obj['lod']
            #     if lod == 1:
            #         roots_3d = obj['roots_3d'][:3]
            #         self.plot_cylinder(ax, 'white', 150, 1000, elevation=roots_3d[2])
            #     elif lod == 2:
            #         roots_3d = obj['roots_3d'][:3]
            #         preds_3d = obj['preds_3d'][:, :3]
            #         age = obj['age']
            #         gender = obj['gender']
            #         if age < 20:
            #             color = 'lightyellow'   # kid
            #         elif gender == 0:
            #             color = 'pink'          # female
            #         else:
            #             color = 'skyblue'       # male
            #         self.plot_joints_and_limbs(ax, preds_3d, roots_3d, color)

            ax.set_xlim([-2000, 2000])
            ax.set_ylim([-2000, 2000])
            ax.set_zlim([0, 2000])
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            fig.canvas.draw()
            fig.canvas.flush_events()
    
    def plot_cylinder(self, ax, color, radius, height, elevation=0, resolution=100):
        theta = np.linspace(0, 2*np.pi, resolution)
        z = np.linspace(elevation, elevation + height, resolution)
        theta, z = np.meshgrid(theta, z)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        ax.plot_surface(x, y, z, alpha=1.0, color=color)
    
    def plot_joints_and_limbs(self, ax, preds_3d, roots_3d, color):
        # joints
        ax.scatter(preds_3d[:, 0], preds_3d[:, 1], preds_3d[:, 2], c=color, s=20)
        
        # root joint
        ax.scatter(roots_3d[0], roots_3d[1], roots_3d[2], c=color, s=100, edgecolor='k', marker='o')
        
        # limbs
        for limb in LIMBS:
            start, end = limb
            ax.plot([preds_3d[start, 0], preds_3d[end, 0]], 
                    [preds_3d[start, 1], preds_3d[end, 1]], 
                    [preds_3d[start, 2], preds_3d[end, 2]], color=color)

class StrategyIsaacSim(VisualizerStrategy):
    def __init__(self, config):
        self.producer = KafkaProducer(server=config.KAFKA.SERVER, topic=config.KAFKA.VISUALIZATION.TOPIC)

    def visualize(self, obj_info):
        self.producer.produce_message(obj_info)

class Visualizer:
    def __init__(self, strategy: VisualizerStrategy):
        self.strategy = strategy
    
    def visualize(self, obj_info):
        self.strategy.visualize(obj_info)

if __name__ == '__main__':
    import time
    visualizer = Visualizer(StrategyPlt3d())
    while True:
        time.sleep(1/15)
        obj_info = [
            {
                'age': 20,
                'gender': 0,
                'lod': 2,
                'preds_3d': np.random.rand(15, 5) * 4000 - 2000,
                'roots_3d': np.random.rand(3) * 4000 - 2000
            },
            {
                'age': 30,
                'gender': 1,
                'lod': 2,
                'preds_3d': np.random.rand(15, 5) * 4000 - 2000,
                'roots_3d': np.random.rand(3) * 4000 - 2000
            },
            {
                'lod': 1,
                'roots_3d': np.random.rand(3) * 4000 - 2000
            }
        ]
        
        visualizer.visualize(obj_info)
