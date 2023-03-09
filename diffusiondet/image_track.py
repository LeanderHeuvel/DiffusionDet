import matplotlib.pyplot as plt
import os 
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, ImageList, Instances
import imageio

class ImageTrack:
    def __init__(self, path, datasets_info):
        self.path = path
        self.threshold = 0.1
        self.metadata = MetadataCatalog.get(
            datasets_info[0] if len(datasets_info) else "__unused"
        )
        self.samplestep = 0
        self.instances = []
        self.cpu_device = torch.device("cpu")
        self.img_data = []

    def load_img(self):
        return read_image(self.path, format="BGR")
        
    def next_samplestep(self):
        self.samplestep += 1
    
    def record_instance(self, instance, samplestep):
        self.instances.append((instance, samplestep))

    def threshold_scores(self, instance:Instances):
        keep_dim = instance.scores > self.threshold
        thresholded = Instances(instance.image_size)
        thresholded.pred_boxes = instance.pred_boxes[keep_dim]
        thresholded.scores = instance.scores[keep_dim]
        thresholded.pred_classes = instance.pred_classes[keep_dim]
        return thresholded.to(self.cpu_device)
    
    def create_gif(self):
        name = os.path.splitext(self.path)[0]
        if self.img_data is None or len(self.img_data) ==0:
            self.generate_imgs()
        filename = name + '_output.gif'
        imageio.mimsave(filename,self.img_data,duration=0.8)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def generate_imgs(self):
        for instance, samplestep in self.instances:
            thresholded = self.threshold_scores(instance) 
            vis_img = Visualizer(self.load_img(), self.metadata).draw_instance_predictions(predictions=thresholded)
            self.img_data.append(vis_img.get_image())

    def save_imgs(self):
        name = os.path.splitext(self.path)[0]
        basepath = name + '_output' 
        for instance, samplestep in self.instances:
            thresholded = self.threshold_scores(instance)
            vis_output = Visualizer(self.load_img(), self.metadata).draw_instance_predictions(predictions=thresholded)
            filename = basepath + '_' + str(samplestep) + '.jpg'
            vis_output.save(filename)

    def print_summed_scores(self):
        for instance, samplestep in self.instances:
            print("timestep: ",samplestep,"summed score: ", torch.sum(instance.scores))

    def __str__(self) -> str:
        return str(self.record)

    def __len__(self) -> int:
        return len(self.record)
