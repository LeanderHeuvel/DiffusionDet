import matplotlib.pyplot as plt
import numpy as np
import os 
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
import imageio
import cv2

class ImageTrack:
    def __init__(self, path, datasets_info, threshold=0.0):
        self.path = path
        self.threshold = threshold
        self.metadata = MetadataCatalog.get(
            datasets_info[0] if len(datasets_info) else "__unused"
        )
        self.samplestep = 0
        self.instances = []
        self.cpu_device = torch.device("cpu")
        self.img_data = []
        self.image_size = None
        self.instances_vectors = []

    def load_img(self):
        return read_image(self.path, format="BGR")
        
    def next_samplestep(self):
        self.samplestep += 1
    
    def record_instance(self, instance, samplestep):
        if self.image_size is None:
            self.image_size = instance.image_size
        self.instances.append((instance, samplestep))
        print("appended instance")
        print(len(self.instances))
    
    def record_vector_instance(self, instance_rand, instance_pred, samplestep):
        if self.image_size is None:
            self.image_size = instance_pred.image_size
        self.instances_vectors.append((instance_rand,instance_pred,samplestep))

    def threshold_scores(self, instance:Instances, keep_dim = None):
        if keep_dim is None:
            keep_dim = instance.scores > self.threshold
        thresholded = Instances(instance.image_size)
        thresholded.pred_boxes = instance.pred_boxes[keep_dim,:]
        thresholded.scores = instance.scores[keep_dim]
        thresholded.pred_classes = instance.pred_classes[keep_dim]
        return thresholded.to(self.cpu_device)
    def get_keep_dim(self, instance:Instances):
        return instance.scores > self.threshold
    
    def create_gif(self, draw_vectors):
        name = os.path.splitext(self.path)[0]
        if self.img_data is None or len(self.img_data) == 0:
            self.generate_imgs(draw_vectors)
        filename = name + '_output_0_' + str(self.threshold) +'.gif'
        imageio.mimsave(filename,self.img_data,duration=1)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def generate_imgs(self, draw_vectors=False):
        if draw_vectors:
            self.generate_vectors()
        else:
            self.generate_bounding_boxes()

    def generate_vectors(self):
        self.img_data = []
        for instance_rand,instance_pred, samplestep in self.instances_vectors:
            keep_dim = self.get_keep_dim(instance_pred)
            # self.set_threshold(0.5) #disable thresholding
            instance_rand_t = self.threshold_scores(instance_rand)
            instance_pred_t = self.threshold_scores(instance_pred)
            vis_img = Visualizer(self.load_img(), self.metadata).draw_instance_predictions(predictions=instance_rand_t,predictions_next=instance_pred_t, draw_vectors=[1,0])
            self.img_data.append(vis_img.get_image())
        '''
            1. sum scores over time using aggregate_scores, i.e. take mean over average 
            2. Normalize values from 0-256
        '''
    def generate_heatmaps(self, aggregate_score = 'mean'):
        img = cv2.imread(self.path)
        heatmap = np.zeros(img.shape)
        # instance_pred: absolute coords, x1, y1, x2, y2
        max_step = 0
        for _, instance_pred, samplestep in self.instances_vectors:
            boxes = instance_pred.pred_boxes
            scores = instance_pred.scores.cpu().numpy()
            boxes = self._convert_boxes(boxes.to(torch.device("cpu")))
            for idx, box in enumerate(boxes):
                box = [int(i) for i in box] 
                # assumes box contains coordinates in x0, y0, x1, y1 format. 
                heatmap[box[1]:box[3], box[0]:box[2], 0] += np.full((abs(box[1]-box[3]), abs(box[0]-box[2])),scores[idx])
                heatmap[box[1]:box[3], box[0]:box[2], 1] += np.full((abs(box[1]-box[3]), abs(box[0]-box[2])),scores[idx])
                heatmap[box[1]:box[3], box[0]:box[2], 2] += np.full((abs(box[1]-box[3]), abs(box[0]-box[2])),scores[idx])
            max_step += 1
        heatmap = self._normalize(heatmap)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(heatmap_img,0.5, img,0.5,0)
        cv2.imwrite("test_heatmap.jpg", superimposed)

    def _normalize(self, heatmap):
        # heatmap = np.log(heatmap) # enable if we want to visualize on log-scale. 
        max_value = max(heatmap.flatten())
        return heatmap/(max_value/254)
    
    def _calculate_mean(self): #DONE
        heatmap = np.zeros(cv2.imread(self.path).shape)
        for instance_pred, _ in self.instances:
            # instance_pred = self.threshold_scores(instance_pred)
            boxes = instance_pred.pred_boxes
            scores = instance_pred.scores.cpu().numpy()
            boxes = self._convert_boxes(boxes.to(torch.device("cpu")))
            for idx, box in enumerate(boxes):
                box = [int(i) for i in box]
                heatmap[box[1]:box[3], box[0]:box[2], :] += np.full((abs(box[1]-box[3]), abs(box[0]-box[2]), 3), scores[idx])
        return heatmap/len(self.instances)

    def _calculate_sd(self):#DONE
        img_shape = cv2.imread(self.path).shape
        mean = self._calculate_mean()
        sum_squared = np.zeros(img_shape)
        for instance_pred, _ in self.instances:
            # instance_pred = self.threshold_scores(instance_pred)
            boxes = instance_pred.pred_boxes
            scores = instance_pred.scores.cpu().numpy()
            boxes = self._convert_boxes(boxes.to(torch.device("cpu")))
            x = np.zeros(img_shape)
            for idx, box in enumerate(boxes):
                box = [int(i) for i in box]
                x[box[1]:box[3], box[0]:box[2], :] += np.full((abs(box[1]-box[3]), abs(box[0]-box[2]),3), scores[idx])
            sum_squared += (x - mean)**2
        return np.sqrt(sum_squared/(len(self.instances)-1))
    '''
    plots a heatmap using cv2.
    Args:
    heatmap: uint8 array that corresponds in dimensions with the image from self.path.
    '''
    def _plot_heatmap(self, heatmap, name="heatmap_plot"):
        img = cv2.imread(self.path)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(heatmap_img,0.5, img,0.5,0)
        filename = name + '.jpg'
        cv2.imwrite(filename, superimposed)

    def generate_analysis(self, name = "plot", measure="mean",): #DONE
        if measure == "mean":
            mean = self._calculate_mean()
            mean = self._normalize(mean)
            self._plot_heatmap(mean,name=name+measure+"_heatmap_plot")
        if measure == "sd":
            sd = self._calculate_sd()
            sd = self._normalize(np.array(sd))
            np.savetxt("heatmap_"+name+measure, sd[:,:,0])
            self._plot_heatmap(sd,name=name+measure+"_heatmap_plot")
        
    def generate_bounding_boxes(self):
        self.img_data = []
        for instance, samplestep in self.instances:
            thresholded = self.threshold_scores(instance) 
            vis_img = Visualizer(self.load_img(), self.metadata).draw_instance_predictions(predictions=thresholded, draw_vectors=[0,0])
            self.img_data.append(vis_img.get_image())
            
    def save_imgs(self,draw_vectors = [0,0]):#TODO: adapt for img_data
        name = os.path.splitext(self.path)[0]
        basepath = name + '_output' 
        for instance, samplestep in self.instances:
            thresholded = self.threshold_scores(instance)
            vis_output = Visualizer(self.load_img(), self.metadata).draw_instance_predictions(predictions=thresholded, draw_vectors=[1,0])
            filename = basepath + '_' + str(samplestep) + '.jpg'
            vis_output.save(filename)

    def print_summed_scores(self):
        for instance, samplestep in self.instances:
            print("timestep: ",samplestep,"summed score: ", torch.sum(instance.scores))

    def __str__(self) -> str: # TODO
        pass

    def __len__(self) -> int: #TODO
        pass
    
    def _convert_boxes(self, boxes):
        """
        Copied from detectrion visualizer.py
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.detach().numpy()
        else:
            return np.asarray(boxes)
