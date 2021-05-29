from .text_detect.file_utils import get_files as _get_files
from PIL import ImageDraw, Image, ImageFont
import numpy as np
import torchvision
from collections import OrderedDict
import json
import os


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def get_args_impl(path, file_name):
    """
    Loads the json file containing the detection/recognition params from parent_path and returns it
    """
    args = {}
    full_path = os.path.join(path, file_name)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            args = json.load(f)
    return args



def get_files_impl(path):
    imgs = []
    if (os.path.isdir(path)):
        imgs, masks, xmls = _get_files(path)
    return imgs


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def draw_fiducials(batch, fiducials):

  # font = ImageFont.truetype("sans-serif.ttf", 16)
  x = batch.clone().detach()
  N = x.shape[0]
  fiducials_ = fiducials.clone().detach()
  for i in range(0, N):
    # need copy because div and sub are in-place operations
    im = torchvision.transforms.ToPILImage()(x[i].div_(2).sub_(-0.5))
    w, h = im.size
    offset_x = 20
    offset_y = 20
    new_w = w + 2 * offset_x
    new_h = h + 2 * offset_y
    canvas = Image.new(im.mode, (new_w, new_h), (255, 255, 255))
    canvas.paste(im, (offset_x, offset_y))
    d = ImageDraw.Draw(canvas)
    fiducials__ = np.dot((fiducials_[i, :, :]), [[w, 0], [0, h]])

    for j in range(0, len(fiducials__)):
      x = fiducials__[j][0] + offset_x
      y = fiducials__[j][1] + offset_y
      # if (0 <= x < w) and (0 <= y < h):
      d.text((x, y), "*", (255, 0, 0))

    return canvas
    # plt.imshow(canvas)
    # plt.show()