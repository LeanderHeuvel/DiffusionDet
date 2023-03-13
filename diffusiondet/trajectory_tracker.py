import matplotlib.pyplot as plt
import numpy as np
from diffusiondet.image_track import ImageTrack

'''
The TrajectoryTracker can keep track of boxes and coordinates and plot these
'''

class TrajectoryTracker:

    def __init__(self, meta_data, lazyload_img=False) -> None:
        self.lazyload_img = lazyload_img
        self.meta_data = meta_data
        self.store = {}

    def record_instance(self, path, instance, time):
        if path not in self.store.keys():
            img_track = ImageTrack(path, self.meta_data)
            self.store[path] = img_track
        self.store[path].record_instance(instance, self.store[path].samplestep)
        self.store[path].next_samplestep()

    def record_vector_instance(self, path, instance_rand, instance_pred, time):
        if path not in self.store.keys():
            img_track = ImageTrack(path, self.meta_data)
            self.store[path] = img_track
        self.store[path].record_vector_instance(instance_rand, instance_pred, self.store[path].samplestep)
        self.store[path].next_samplestep()
    


    def store_trajectory(self):
        store_img = np.frompyfunc(lambda x: x.save_imgs(), 1,0)
        store_img(list(self.store.values()))
    
    def create_gifs(self, draw_vectors):
        store_img = np.frompyfunc(lambda x: x.create_gif(draw_vectors), 1,0)
        store_img(list(self.store.values()))

    def print_summed_scores(self):
        store_img = np.frompyfunc(lambda x: x.print_summed_scores(), 1,0)
        store_img(list(self.store.values()))

    def __str__(self) -> str:
        return str(self.store)
    
    def __sizeof__(self) -> int:
        return len(self.store)