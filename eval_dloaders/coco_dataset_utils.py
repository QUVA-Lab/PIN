# Copyright (c) Facebook, Inc. and its affiliates, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only
# https://apache.org/licenses/LICENSE-2.0.txt
# Not a Contribution

from torchvision.transforms import functional as F
import numpy as np

coco_mapping = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18,
    20:19, 21:20, 22:21, 23:22, 24:23, 25:24, 27:25, 28:26, 31:27, 32:28, 33:29, 34:30, 35:31, 36:32, 37:33, 38:34
    , 39:35, 40:36, 41:37, 42:38, 43:39, 44:40, 46:41, 47:42, 48:43, 49:44, 50:45, 51:46, 52:47, 53:48, 54:49, 55:50
    , 56:51, 57:52, 58:53, 59:54, 60:55, 61:56, 62:57, 63:58, 64:59, 65:60, 67:61, 70:62, 72:63, 73:64, 74:65, 75:66
    , 76:67, 77:68, 78:69, 79:70, 80:71, 81:72, 82:73, 84:74, 85:75, 86:76, 87:77, 88:78, 89:79, 90:80
}

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
    {"synset": "dog.n.01", "coco_cat_id": 18},
    {"synset": "horse.n.01", "coco_cat_id": 19},
    {"synset": "sheep.n.01", "coco_cat_id": 20},
    {"synset": "beef.n.01", "coco_cat_id": 21},
    {"synset": "elephant.n.01", "coco_cat_id": 22},
    {"synset": "bear.n.01", "coco_cat_id": 23},
    {"synset": "zebra.n.01", "coco_cat_id": 24},
    {"synset": "giraffe.n.01", "coco_cat_id": 25},
    {"synset": "backpack.n.01", "coco_cat_id": 27},
    {"synset": "umbrella.n.01", "coco_cat_id": 28},
    {"synset": "handbag.n.04", "coco_cat_id": 31},
    {"synset": "necktie.n.01", "coco_cat_id": 32},
    {"synset": "tote_bag.n.06", "coco_cat_id": 33},
    {"synset": "frisbee.n.01", "coco_cat_id": 34},
    {"synset": "ski.n.01", "coco_cat_id": 35},
    {"synset": "snowboard.n.01", "coco_cat_id": 36},
    {"synset": "ball.n.06", "coco_cat_id": 37},
    {"synset": "kite.n.03", "coco_cat_id": 38},
    {"synset": "baseball_bat.n.01", "coco_cat_id": 39},
    {"synset": "baseball_glove.n.01", "coco_cat_id": 40},
    {"synset": "skateboard.n.01", "coco_cat_id": 41},
    {"synset": "surfboard.n.01", "coco_cat_id": 42},
    {"synset": "tennis_racket.n.01", "coco_cat_id": 43},
    {"synset": "bottle.n.01", "coco_cat_id": 44},
    {"synset": "wineglass.n.01", "coco_cat_id": 46},
    {"synset": "cup.n.01", "coco_cat_id": 47},
    {"synset": "fork.n.01", "coco_cat_id": 48},
    {"synset": "knife.n.01", "coco_cat_id": 49},
    {"synset": "spoon.n.01", "coco_cat_id": 50},
    {"synset": "bowl.n.03", "coco_cat_id": 51},
    {"synset": "banana.n.02", "coco_cat_id": 52},
    {"synset": "apple.n.01", "coco_cat_id": 53},
    {"synset": "sandwich.n.01", "coco_cat_id": 54},
    {"synset": "orange.n.01", "coco_cat_id": 55},
    {"synset": "broccoli.n.01", "coco_cat_id": 56},
    {"synset": "carrot.n.01", "coco_cat_id": 57},
    {"synset": "sausage.n.02", "coco_cat_id": 58},
    {"synset": "pizza.n.01", "coco_cat_id": 59},
    {"synset": "doughnut.n.02", "coco_cat_id": 60},
    {"synset": "cake.n.03", "coco_cat_id": 61},
    {"synset": "chair.n.01", "coco_cat_id": 62},
    {"synset": "sofa.n.01", "coco_cat_id": 63},
    {"synset": "pot.n.04", "coco_cat_id": 64},
    {"synset": "bed.n.01", "coco_cat_id": 65},
    {"synset": "dining_table.n.01", "coco_cat_id": 67},
    {"synset": "toilet.n.02", "coco_cat_id": 70},
    {"synset": "television_set.n.01", "coco_cat_id": 72},
    {"synset": "laptop_computer.n.01", "coco_cat_id": 73},
    {"synset": "mouse.n.04", "coco_cat_id": 74},
    {"synset": "remote_control.n.01", "coco_cat_id": 75},
    {"synset": "computer_keyboard.n.01", "coco_cat_id": 76},
    {"synset": "cellular_telephone.n.01", "coco_cat_id": 77},
    {"synset": "microwave_oven.n.02", "coco_cat_id": 78},
    {"synset": "oven.n.01", "coco_cat_id": 79},
    {"synset": "toaster.n.02", "coco_cat_id": 80},
    {"synset": "sink.n.01", "coco_cat_id": 81},
    {"synset": "refrigerator.n.01", "coco_cat_id": 82},
    {"synset": "book.n.01", "coco_cat_id": 84},
    {"synset": "clock.n.01", "coco_cat_id": 85},
    {"synset": "vase.n.01", "coco_cat_id": 86},
    {"synset": "scissors.n.01", "coco_cat_id": 87},
    {"synset": "teddy_bear.n.01", "coco_cat_id": 88},
    {"synset": "hair_dryer.n.01", "coco_cat_id": 89},
    {"synset": "toothbrush.n.01", "coco_cat_id": 90},
]


class Resize(object):
    def __init__(self, size):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, img, label):
        w_ratio = self.size[0] / img.width
        h_ratio = self.size[1] / img.height

        adjusted_annos = []
        for anno in label:
            adjusted_bbox = [int(anno['bbox'][0] * w_ratio), int(anno['bbox'][1] * h_ratio), int(anno['bbox'][2] * w_ratio), int(anno['bbox'][3] * h_ratio)]
            adjusted_annos.append({'bbox': adjusted_bbox, 'category_id': anno['category_id']})

        return F.resize(img, self.size), adjusted_annos

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        return format_string

def coco_get_objects_names(label):
    """
    Extracts a list of object names from the annotations in a COCO-format label.
    :param label: A list of dictionaries, each representing an annotation in COCO format.
    :return: A list containing the names of objects corresponding to the 'category_id' in each annotation.
    """
    object_names = []
    for annotation in label:
        category_id = coco_mapping[annotation['category_id']] -1
        object_name = COCO_CLASSES[category_id]
        object_names.append(object_name)
    return object_names

def coco_get_objects_ids(label):
    """
    Extracts a list of object category IDs from the annotations in a COCO-format label.
    :param label: A list of dictionaries, each representing an annotation in COCO format.
    :return: A list containing the category IDs for each object in the annotations.
    """
    ids = []
    for annotation in label:
        category_id = coco_mapping[annotation['category_id']] - 1
        ids.append(category_id)
    return ids

def coco_get_bboxes(label, image_size=None):
    """
    Obtains a list of bounding boxes from COCO-format annotations.
    :param label: A list of dictionaries, each representing an annotation in COCO format.
    :param image_size: Optional tuple (width, height) specifying the dimensions to clip bounding box coordinates.
    :return: A list of bounding boxes, potentially clipped to the provided image dimensions.
    """
    if image_size is None:
        width, height = 224, 224
    else:
        width, height = image_size
    bboxes_list = [[annotation['bbox'][0], annotation['bbox'][1], min(annotation['bbox'][0]+annotation['bbox'][2], width-1), min(annotation['bbox'][1]+annotation['bbox'][3], height-1)] for annotation in label]
    return bboxes_list

def coco_bboxes_to_gc(bboxes, grid_size, image_size=None):
    """
    Converts bounding boxes to grid cell coordinates based on the specified grid size and image dimensions.
    :param bboxes: A list of bounding boxes.
    :param grid_size: The size of the grid.
    :param image_size: Optional tuple (width, height) specifying the image dimensions.
    :return: A list of grid coordinates corresponding to each bounding box.
    """
    if image_size is None:
        width, height = 224, 224
    else:
        width, height = image_size
    grid_width, grid_height = np.ceil([width / grid_size, height / grid_size])
    grid_coordinates = [[ int(box[0]/grid_width), int(box[1]/grid_height), int(box[2]/grid_width), int(box[3]/grid_height)] for box in bboxes]
    return grid_coordinates


def coco_gc_to_bboxes(grid_coordinates, grid_size, image_size=None):
    """
    Converts grid cell coordinates back to bounding boxes based on the specified grid size and image dimensions.
    :param grid_coordinates: A list of grid cell coordinates.
    :param grid_size: The size of the grid.
    :param image_size: Optional tuple (width, height) specifying the image dimensions.
    :return: A list of bounding boxes derived from the grid coordinates.
    """
    if image_size is None:
        width, height = 224, 224
    else:
        width, height = image_size
    grid_width, grid_height = np.ceil([width / grid_size, height / grid_size])
    bboxes = [[(box[0])*grid_width, (box[1])*grid_height,
                       min((box[2]+1)*grid_width-1, width-1), min((box[3]+1)*grid_height-1, height-1)]
                      for box in grid_coordinates]
    return bboxes

def coco_get_gc(label, grid_size, image_size=None):
    """
    Computes grid cell coordinates for bounding boxes obtained from COCO-format annotations.
    :param label: A list of dictionaries, each representing an annotation in COCO format.
    :param grid_size: The size of the grid.
    :param image_size: Optional tuple (width, height) specifying the image dimensions.
    :return: A list of grid coordinates for each bounding box.
    """
    bboxes = coco_get_bboxes(label, image_size)
    return coco_bboxes_to_gc(bboxes, grid_size, image_size)

