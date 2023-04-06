import cv2
import os


def dataset_mapper(img_path = 'data/VisDrone2019-DET-train/images/',annotations_path = 'data/VisDrone2019-DET-train/annotations/'):
    ls = []
    keep = [1,4,9,6] # keep: pedestrian, car, bus, truck
    for id, filename in enumerate(os.listdir(img_path)[:100]):
        image  = {}
        image['file_name'] = img_path + filename
        img = cv2.imread(image['file_name'])
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['image_id'] = id
        filename.split('.')[0]
        instances = []
        with open(annotations_path+filename.split('.')[0]+'.txt', 'r') as f:
            for line in f.readlines():
                instance = {}
                coords = line.split(',')[:4]
                instance['bbox'] = list(map(int, coords))
                instance['bbox_mode'] = 1
                instance['category_id'] = int(line.split(',')[5])
                if instance['category_id'] == 2: # convert people into pedestrian
                    instance['category_id'] = 1
                if instance['category_id'] in keep:
                    instances.append(instance)
        image['annotations'] = instances
        ls.append(image)
    return ls