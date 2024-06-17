import numpy as np
from torchvision.transforms import functional as F


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
VOC_CLASS2IND = {class_name: index for index, class_name in enumerate(VOC_CLASSES)}

class Resize(object):
    def __init__(self, size):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, img, label):
        w_ratio = self.size[0] / img.width
        h_ratio = self.size[1] / img.height
        label['annotation']['size']['width'] = self.size[0]
        label['annotation']['size']['height'] = self.size[1]

        if isinstance(label['annotation']['object'], list):
            for i in range(len(label['annotation']['object'])):
                label['annotation']['object'][i]['bndbox']['xmin'] = np.clip(int(int(label['annotation']['object'][i]['bndbox']['xmin']) * w_ratio), 1, self.size[0])
                label['annotation']['object'][i]['bndbox']['xmax'] = np.clip(int(int(label['annotation']['object'][i]['bndbox']['xmax']) * w_ratio), 1, self.size[0])
                label['annotation']['object'][i]['bndbox']['ymin'] = np.clip(int(int(label['annotation']['object'][i]['bndbox']['ymin']) * h_ratio), 1, self.size[1])
                label['annotation']['object'][i]['bndbox']['ymax'] = np.clip(int(int(label['annotation']['object'][i]['bndbox']['ymax']) * h_ratio), 1, self.size[1])
        else:
            label['annotation']['object']['bndbox']['xmin'] = np.clip(int(int(label['annotation']['object']['bndbox']['xmin']) * w_ratio), 1, self.size[0])
            label['annotation']['object']['bndbox']['xmax'] = np.clip(int(int(label['annotation']['object']['bndbox']['xmax']) * w_ratio), 1, self.size[0])
            label['annotation']['object']['bndbox']['ymin'] = np.clip(int(int(label['annotation']['object']['bndbox']['ymin']) * h_ratio), 1, self.size[1])
            label['annotation']['object']['bndbox']['ymax'] = np.clip(int(int(label['annotation']['object']['bndbox']['ymax']) * h_ratio), 1, self.size[1])
        return F.resize(img, self.size), label
    
def voc_get_objects_names(label):
    """
    Parses a VOC detection label to obtain a list of the names of classes corresponding to objects detected in an image.
    :param label: A label in the VOC detection format for an image.
    :return: A list containing the names of classes for objects detected in the image.
    """
    objects = label['annotation']['object']
    if isinstance(objects, list):
        names = [o['name'] for o in objects]
    else:
        names = [objects['name']]
    return names


def voc_get_objects_ids(label):
    """
    Extracts a list of ids of classes appear in the image from a VOC detection label
    :param label: A label in the VOC detection format for an image.
    :return: list of ids of classes appear in the image
    """
    names = voc_get_objects_names(label)
    return [VOC_CLASS2IND[name] for name in names]


def voc_get_bboxes(label):
    """
    Obtains a list of bounding boxes from a label in VOC detection format.
    :param label: A label of an image in VOC detection format.
    :return: A list of bounding boxes.
    """
    objects = label['annotation']['object']
    if isinstance(objects, list):
        bboxes_list = [[int(o['bndbox'][coord]) for coord in ['xmin', 'ymin', 'xmax', 'ymax']] for o in objects]
    else:
        bboxes_list = [[int(objects['bndbox'][coord]) for coord in objects['bndbox']]]
    return bboxes_list


def voc_bboxes_to_gc(bboxes, image_size, grid_size):
    """
    Converts a list of bounding boxes into corresponding grid coordinates.
    :param bboxes: A list of bounding boxes.
    :param image_size: A dictionary containing the image's width and height.
    :param grid_size: The size of the grid.
    :return: A list of grid coordinates for each bounding box, specifying the top left and bottom right cells.
    """
    width, height = int(image_size['width']), int(image_size['height'])
    grid_width, grid_height = np.ceil([width / grid_size, height / grid_size])
    grid_coordinates = [[int((box[0]-1)/grid_width), int((box[1]-1)/grid_height),
                         int((box[2]-1)/grid_width), int((box[3]-1)/grid_height)]
                        for box in bboxes]
    return grid_coordinates


def voc_gc_to_bboxes(grid_coordinates, image_size, grid_size):
    """
    Converts list of grid locations to a list of bounding boxes in the center of each grid cell.
    :param grid_coordinates: list of grid coordinates for objects
    :param image_size: a dictionary with the width and height of an image
    :param grid_size: the grid size
    :return: A list of the grid location for each upper left and bottom right pixel locations
    """
    width, height = int(image_size['width']), int(image_size['height'])
    grid_width, grid_height = np.ceil([width / grid_size, height / grid_size])
    bboxes = [[(box[0])*grid_width, (box[1])*grid_height,
                       min((box[2]+1)*grid_width-1, width-1), min((box[3]+1)*grid_height-1, height-1)]
                      for box in grid_coordinates]
    return bboxes


def voc_get_gc(label, grid_size):
    """
    This function calculates the grid cell coordinates for bounding boxes obtained from a given label.
    :param label: A label of an image in VOC detection format
    :param grid_size: the grid size
    :return: A list of grid coordinates for each bounding box, specifying the top left and bottom right cells.
    """
    bboxes = voc_get_bboxes(label)
    gc = voc_bboxes_to_gc(bboxes, label['annotation']['size'], grid_size)
    return gc