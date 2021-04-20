from typing import Dict, TypedDict
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from .utils import Map, copyStateDict
import cv2
from .text_detect.craft import CRAFT
from .text_detect import craft_utils
from .text_detect import imgproc
from PIL.Image import Image

# Necessary for pytorch to work in a multi-process mode
torch.set_num_threads(1)


class TextDetectRet(TypedDict):
    bboxes: Dict[str, str]
    region_score_map: Image
    affinity_score_map: Image
    connected_component_mask: Image


def init_detection_model(args: Dict) -> CRAFT:
    """ Initializes the text detection model from the input parameters passed as the argument. The input parameters
    dictionary must contain a DET_MODEL_PATH field, which points to the location of the pre-trained text detection
    model
    """
    args = Map(args)
    if not os.path.exists(args.trained_model):
        raise ValueError("Incorrect path for text detection model")
    net = CRAFT()  # initialize

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    return net


def text_detect(img: np.array, net: CRAFT, args: Dict, logger=None) \
        -> TextDetectRet:
    """
    Takes as input an 3-channel numpy array of uint8 values in range [0-255] (rgb image), the CRAFT text detection
    neural network model (returned by init_detection_model), and detection parameters and returns a dictionary with
    the following structure:
    bboxes: a dictionary consisting of key-value pairs, where key is crop image name and value is corresponding
    bounding box coordinates. For example, if there are 5 detected words in an input image, then bboxes would
    look like:
    'bboxes': { 'crop0': comma-seperated bounding box coordinates,
                'crop1': comma-seperated bounding box coordinates,
                ...
                'crop4': comma-seperated bounding box coordinates
              }
    'region_score_map': PIL Image that encodes the probability of each pixel being the center of a character
    (for diagnostics only)

    'affinity_score_map': PIL Image that encodes the probability of each pixel being the center of region *between*
    two characters (for diagnostics only)

    'connected_component_mask': PIL Image that contains a mask that shows connected components in an image. Ideally
    these connected components correspond to distinct words (for diagnostics only).
    If the input arguments are of incorrect types/shapes, a ValueError exception is raised. If the input image pixels
    are not np.uint8, a TypeError exception is raised.
    """

    # Check for argument types
    type_ok: bool = isinstance(img, np.ndarray) and len(img.shape) == 3 and isinstance(net, CRAFT) \
                    and isinstance(args, Dict)
    if not type_ok:
        raise ValueError("Incorrect argument types passed to text detection")

    # image pixels must be uint8
    if img.dtype != np.uint8:
        raise TypeError("Image pixels must be uint8")

    # check if elements of the input image are between 0 and 255 (actually redundant because pixels are uint8)
    if img.min() < 0 or img.max() > 255:
        raise ValueError("Input image pixel values must be between [0, 255]")

    try:
        args = Map(args)
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, args.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=args.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        args.ratio_h = ratio_h
        args.ratio_w = ratio_w
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if args.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        refine_net = None
        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        boxes, polys, cc_mask = craft_utils.getDetBoxes(score_text, score_link, args.text_threshold,
                                                        args.link_threshold,
                                                        args.low_text, args.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, args.ratio_w, args.ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, args.ratio_w, args.ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # region score
        region_score_hm = imgproc.cvt2HeatmapImg(score_text)

        # affinity score
        affinity_score_hm = imgproc.cvt2HeatmapImg(score_link)

        # connected components
        cc_mask_hm = imgproc.cvt2HeatmapImg(cc_mask)

        bbox_crop_map = {}
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            bbox_coords_str = ','.join([str(p) for p in poly]) + '\r\n'
            x_coords = poly[0:-1:2]
            y_coords = poly[1::2]
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            crop_index = "crop{0}".format(i)
            bbox_crop_map[crop_index] = bbox_coords_str

        return {'bboxes': bbox_crop_map, 'region_score_map': region_score_hm, 'affinity_score_map': affinity_score_hm,
                'connected_component_mask': cc_mask_hm}
    except Exception as e:
        if logger:
            logger.exception(e)
        raise
