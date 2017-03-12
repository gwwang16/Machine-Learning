
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


def fetch_csv_data(filepath, delimiter=',', consider_only_a_sample=False, univ_new_line=False,
                   include_only_these_fields=None, clean_up_field_names=False,
                   unique_index_fields=None):    
    assert os.path.isfile(filepath)
    data_raw = []

    open_flag = 'rb'
    open_flag += 'U' if univ_new_line else ''
    row_counter = 0

    with open(filepath, open_flag) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        fields = None
        
        for row in reader:
            row_counter += 1
            #if row_counter % 1000 == 0: print 'loaded {} rows'.format(row_counter)
            
            assert len(row) > 1
            if fields is None:
                fields = row
                if clean_up_field_names:
                    fields = [f.replace(' ', '_').lower() for f in fields]
                continue

            if len(fields) != len(row):
                print ('fields:', fields)
                print ('row:', row)
                                    
            assert len(fields) == len(row)

            # remove fields not in 'include_only_these_fields' if it's defined
            if include_only_these_fields is None:
                d = OrderedDict(zip(fields, row))
            else:
                assert set(include_only_these_fields).issubset(set(fields))
                d = OrderedDict()
                for i, f in enumerate(fields):
                    if f in include_only_these_fields:
                        d[f] = row[i]
            
            data_raw.append(d)

    types = determine_types_from_rows(data_raw, consider_only_a_sample)

    data = apply_types_to_rows(types, data_raw)

    if unique_index_fields is not None:
        data = add_unique_index_to_row_of_dicts(data, unique_index_fields)

    return data    


def cv2_goto_frame(cap, frame_id):
	return cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)


def display_states(dataframe,frame_id):
	'''display sample of the processed dataset'''
	sample_img = dataframe.imgs[frame_id]
	plt.imshow(sample_img)


def load_batch(dataframe, batch_size, batch_i):
    #imgs = dataframe['imgs']
    #wheels = dataframe['wheels']
    #n = len(imgs)

    #batch_count = int(np.ceil(frame_count / batch_size))
    #for batch in range(batch_count):
    df = pd.DataFrame()
    frame_start = batch_i * batch_size
    frame_end = frame_start + batch_size
    df = dataframe[frame_start:frame_end]
   
    return df

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