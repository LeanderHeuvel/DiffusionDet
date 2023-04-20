import cv2
import os
import json

limit = 1000

def dataset_mapper(img_path = 'data/VisDrone2019-DET-train/images/',annotations_path = 'data/VisDrone2019-DET-train/annotations/'):
    ls = []
    keep = [1,4,9,6] # keep: pedestrian, car, bus, truck
    for id, filename in enumerate(os.listdir(img_path)[:limit]):
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

def coco_dataset_mapper(img_path = 'datasets/coco/val2017/',annotations_path = 'datasets/coco/annotations/instances_val2017.json'):
    ls = []
    f = open(annotations_path)
    js = json.load(f)
    images = sorted(js['images'], key = lambda x: x['id'])
    annotations = sorted(js['annotations'], key = lambda x: x['image_id'])
    an_it = iter(annotations)
    annotation = next(an_it)
    for img in images[0:limit]:
        image = {}
        image['file_name'] = img_path + img['file_name']
        image['height'] = img['height']
        image['width'] = img['width']
        image['image_id'] = img['id']
        instances = []
        while(annotation['image_id']==img['id']):
            instance = {}
            instance['bbox'] = annotation['bbox']
            instance ['bbox_mode'] = 1
            instance['category_id'] = adjust_unused_idxs(annotation['category_id'])
            instances.append(instance)
            annotation = next(an_it)
        image['annotations'] = instances
        ls.append(image)
    return ls

def coco_dataset_mapper_sub(img_path = 'datasets/coco/val2017/',annotations_path = 'datasets/coco/annotations/instances_val2017.json'):
    ls = []
    keep = [0,2,5,7] # keep: person, car, bus, truck
    f = open(annotations_path)
    js = json.load(f)
    images = sorted(js['images'], key = lambda x: x['id'])
    annotations = sorted(js['annotations'], key = lambda x: x['image_id'])
    an_it = iter(annotations)
    annotation = next(an_it)
    for img in images[0:limit]:
        image = {}
        image['file_name'] = img_path + img['file_name']
        image['height'] = img['height']
        image['width'] = img['width']
        image['image_id'] = img['id']
        instances = []
        while(annotation['image_id']==img['id']):
            instance = {}
            instance['bbox'] = annotation['bbox']
            instance ['bbox_mode'] = 1
            instance['category_id'] = adjust_unused_idxs(annotation['category_id'])
            if instance['category_id'] in keep:
                instances.append(instance)
            annotation = next(an_it)
        image['annotations'] = instances
        ls.append(image)
    return ls

def adjust_unused_idxs(category_id, adjust_range = True):
    if adjust_range:
        category_id_new = category_id - 1
    else:
        category_id_new = category_id
    unused = [12,26,29,30,45,66,68,69,71,83,91]
    for un in unused:
        if category_id >= un:
            category_id_new -= 1
        else:
            return category_id_new
    return category_id_new 