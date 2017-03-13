
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import params
import os
import csv


def cv2_current_frame(cap):
	'''return the current frame of this video'''
	return int(cap.get(cv2.CAP_PROP_POS_FRAMES))

def frame_count(file_path):
	'''return frame count of this video'''
	cap = cv2.VideoCapture(file_path)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	return frame_count


def cv2_goto_frame(cap, frame_id):
	return cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)



def load_batch(batch_size, batch_i):
    #imgs = dataframe['imgs']
    #wheels = dataframe['wheels']
    #n = len(imgs)

    #batch_count = int(np.ceil(frame_count / batch_size))
    #for batch in range(batch_count):
    frame_start = batch_i * batch_size
    frame_end = frame_start + batch_size
    return frame_start, frame_end

def load_batch_v2(imgs, wheels):

    assert len(imgs) == len(wheels)
    n = len(imgs)

    assert n > 0

    ii = random.sample(range(0, n), params.batch_size)
    assert len(ii) == params.batch_size

    xx, yy = [], []
    for i in ii:
        xx.append(imgs[i])
        yy.append(wheels[i])

    return xx, yy