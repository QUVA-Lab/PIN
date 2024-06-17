import re
import numpy as np
import torch

class GridCodec():
    def __init__(self, args):
        super().__init__()
        self.grid_size = args.grid_size
        
        self.max_value = args.range_max_value
        self.image_size = args.image_size
        
        self.grid_horizontal_names = self._generate_numbers(args.grid_size)
        self.grid_vertical_names = self._generate_numbers(args.grid_size)
        self.obj_template = "[{grid_hname_1},{grid_vname_1},{grid_hname_2},{grid_vname_2}]"
        self.grid_vertical_indices = {name: i for i, name in enumerate(self.grid_vertical_names)}
        self.grid_horizontal_indices = {name: i for i, name in enumerate(self.grid_horizontal_names)}

    def _generate_numbers(self, grid_size):
        max_cell = self.max_value
        
        # Generate the numbers with desired precision
        numbers = [(max_cell * i )/ (grid_size) for i in range(grid_size + 1)]
        
        formatted_numbers = []
        for num in numbers:
            if num == int(num):  # Check if number is an integer
                formatted_numbers.append(str(int(num)))
            elif int(num * 10) == num * 10:
                formatted_numbers.append("{:.1f}".format(num))
            else:
                formatted_numbers.append("{:.2f}".format(num))
                    
        return formatted_numbers

    def gc_to_bboxes_default_image_size(self, grid_coordinates, grid_size):
        width, height = self.image_size, self.image_size
        grid_width, grid_height = np.ceil([width / grid_size, height / grid_size])
        bboxes = [[(box[0])*grid_width, (box[1])*grid_height,
                        min((box[2]+1)*grid_width, width), min((box[3]+1)*grid_height, height)]
                        for box in grid_coordinates]
        return bboxes
    
    def encode(self, coordinates):
        return self.obj_template.format(grid_vname_1=self.grid_vertical_names[coordinates[1]],
                                        grid_vname_2=self.grid_vertical_names[coordinates[3] + 1],
                                        grid_hname_1=self.grid_horizontal_names[coordinates[0]],
                                        grid_hname_2=self.grid_horizontal_names[coordinates[2] + 1])
    
    def decode(self, text, target=None):
        
        # Assume single sentence!! - if not, how to split by '.' in an way that is not weird
        sentence = text

        # For now assume one object
        obj = sentence
        # Extract the object name
        try:
            obj_n = re.search(f'a (.*) {self.middle_prompt[:2]}', sentence).group(1)   
        except:
            obj_n = ''             # if not match:
        #     continue
        # Extract the grid locations
        match = re.search(r' \[(.+?),(.+?),(.+?),(.+?)\]', obj)
        if not match or len(match.groups()) != 4:
            xmin, ymin, xmax, ymax = 0, 0, 1, 1
        else:
            xmin, ymin, xmax, ymax = match.groups()
            if ymin not in self.grid_vertical_names[:-1] or ymax not in self.grid_vertical_names[1:] \
                    or xmin not in self.grid_horizontal_names[:-1] or xmax not in self.grid_horizontal_names[1:]:
                xmin, ymin, xmax, ymax = 0, 0, 1, 1
            else:
                xmin, ymin = self.grid_horizontal_indices[xmin], self.grid_vertical_indices[ymin]
                xmax, ymax = self.grid_horizontal_indices[xmax], self.grid_vertical_indices[ymax]
                if xmin >= xmax or ymin >= ymax:
                    xmin, ymin, xmax, ymax = 0, 0, 1, 1
        boxes = [[xmin, ymin, xmax-1, ymax-1]]
        boxes = self.gc_to_bboxes_default_image_size(boxes, self.grid_size)
        bboxs = torch.cat(([torch.tensor([boxes[0]])]), axis=0)
        return [obj_n], bboxs
