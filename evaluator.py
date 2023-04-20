from detectron2.evaluation import DatasetEvaluator

from torchvision.ops import box_convert
from torchvision.ops import box_iou
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch import cuda
import torch

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from pprint import pprint


class CustomEvaluator(DatasetEvaluator):
    
    def __init__(self) -> None:
        if cuda.is_available():  
            self.device = "cuda:0" 
        else:  
            self.device = "cpu"  
        super().__init__()
        self.detected_boxes = 0
        self.num_boxes = 0
        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.visdrone_labels = ["ignored_regions","pedestrian", "people","bicyle","car","van", "truck","tricycle","awning-tricyle","bus","motor", "others"]
        self.metric = MeanAveragePrecision()

    def _show(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig("test_img.jpg")

    def reset(self):
        self.metric.reset()
        return super().reset()
    
    def _convert(self, boxes):
        return box_convert(boxes,"xywh","xyxy")
    '''
    inputs: list of dictionaries with keys:
      file_name
      height
      width
      image_id
      image: tensor(3,width, height)
      instances: Instances with fields: gt_boxes, gt_classes
    outputs: list of dictionaries with keys:
      instances: Instances(), with fields: pred_boxes, scores, pred_classes
      
    '''
    

    def process(self, inputs, outputs):
        # img = inputs[0]['image']
        # labels = np.array(self.visdrone_labels)[outputs[0]['instances'].pred_classes.tolist()]
        
        # img = draw_bounding_boxes(img, outputs[0]['instances'].pred_boxes.tensor,labels=labels.tolist(), colors='red', font_size=100)
        # img = draw_bounding_boxes(img, inputs[0]['instances'].gt_boxes.tensor, colors='green')
        # self._show(img)
        gt_boxes = self.convert_to_torchmetric(inputs, pred=False)
        pred_boxes = self.convert_to_torchmetric(outputs, pred=True)
        self.metric.update(preds = pred_boxes, target = gt_boxes)
        return super().process(inputs, outputs)
    '''
    
    '''
    def convert_to_torchmetric(self, instances, pred=True):
        if pred:
            return [{'boxes':instance['instances'].pred_boxes.tensor.to(self.device),'scores':instance['instances'].scores.to(self.device),'labels':instance['instances'].pred_classes.to(self.device)} for instance in instances]
        return [{'boxes':instance['instances'].gt_boxes.tensor.to(self.device),'labels':instance['instances'].gt_classes.to(self.device)} for instance in instances]

    def evaluate(self):
        return self.metric.compute()