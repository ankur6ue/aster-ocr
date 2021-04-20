import os
import torch
from typing import Dict, TypedDict
import torchvision
from PIL.Image import Image as ImageType  # the Type
from PIL import Image  # the module
import numpy as np
from .text_rec.utils.serialization import load_checkpoint
from .text_rec.models.model_builder import ModelBuilder
from .text_rec.evaluation_metrics.metrics import get_str_list
from .text_rec.utils.labelmaps import get_vocabulary
import json
from .utils import Map, draw_fiducials

torch.set_num_threads(1)


class DataInfo(object):
  """
  Save the info about the dataset.
  This a code snippet from dataset.py
  """
  def __init__(self, voc_type):
    super(DataInfo, self).__init__()
    self.voc_type = voc_type

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)


def init_rec_model(args: Dict):
    """
    Initializes the text recognition model from the input parameters passed as the argument. The input parameters
    dictionary must contain a REC_MODEL_PATH field, which points to the location of the pre-trained text recognition
    model
    """
    args = Map(args)
    if not os.path.exists(args.trained_model):
        raise ValueError("Incorrect path for text recognition model")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)

    dataset_info = DataInfo(args.voc_type)

    # Create model
    model = ModelBuilder(dataset_info, args)

    # Load from checkpoint
    model_path = args.trained_model
    model_wgts = load_checkpoint(model_path)
    model.load_state_dict(model_wgts['state_dict'])

    device = 'cpu'
    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        # model = nn.DataParallel(model)

    # Evaluation
    model.eval()
    return {'model': model, 'dataset': dataset_info}


def preprocess(img, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = torchvision.transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img


# Defining type for model_artifact
class TextRecModelParamType(TypedDict):
    model: ModelBuilder
    dataset: DataInfo


class TextRecRetType(TypedDict):
    result: str
    stn_input_with_fid: ImageType
    stn_output: ImageType


def recognize(imgs: Dict[str, np.ndarray], args: Dict, model_artifact: TextRecModelParamType, logger=None)\
        -> Dict[str, TextRecRetType]:
    """
    Takes as input a list of 3-channel numpy array (rgb image), text recognition parameters and the text detection model
    and returns a dictionary with the following structure:
    key: crop_i, where the index i corresponds to the index of the input image list
    value: a dictionary with the following structure:
        'result': recognition result string
        'stn_input_with_fid': PIL Image that is input to the spatial transformation network (STN) used to rectify
        each input image with the fiducials drawn on it (for diagnostics only)
        'stn_output': PIL Image that is the output of STN (for diagnostics only)
    The input to this method is typically the output of the text detection module, which returns the bounding boxes
    for each word detected in an input document image. Image regions for each bounding box are cropped out of the
    input image and inserted into a list. The list can then be passed to this method.
    If the input arguments are of incorrect types/shapes, a ValueError exception is raised. If the input image pixels
    are not np.uint8, a TypeError exception is raised.
    """

    # validate input argument types
    type_ok: bool = isinstance(imgs, dict) and isinstance(model_artifact.get("model"), ModelBuilder) and \
                    isinstance(args, dict)

    if not type_ok:
        raise ValueError("Incorrect argument types passed to text recognition module")

    try:
        args = Map(args)
        rec_results = {}
        samples = 0
        model = model_artifact['model']
        device = 'cpu'
        for crop_name, img in imgs.items():
            # image pixels must be uint8
            if img.dtype != np.uint8:
                raise TypeError("Image pixels must be uint8")

            if not (isinstance(img, np.ndarray) and len(img.shape) == 3 and len(img) > 0):
                raise ValueError("Incorrect image type passed to text recognition module. Input image \
                                 must be a uint8 ndarray of shape=3")

            if img is None:
                continue
            if logger:
                logger.info("performing recognition on {0}".format(crop_name))

            dataset_info = model_artifact['dataset']
            preprocessed_img = preprocess(Image.fromarray(img, "RGB"))
            with torch.no_grad():
                preprocessed_img = preprocessed_img.to(device)
            input_dict = {}
            input_dict['images'] = preprocessed_img.unsqueeze(0)

            # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
            rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
            rec_targets[:, args.max_len - 1] = dataset_info.char2id[dataset_info.EOS]
            input_dict['rec_targets'] = rec_targets
            input_dict['rec_lengths'] = [args.max_len]

            output_dict, stn_input, stn_output, fiducials = model(input_dict)

            samples += 1
            stn_input_with_fid = draw_fiducials(stn_input, fiducials)
            stn_output = torchvision.transforms.ToPILImage()(stn_output[0].div_(2).sub_(-0.5))

            pred_rec = output_dict['output']['pred_rec']
            pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
            rec_result = pred_str[0]

            # print("processing on thread: {0}".format(threading.get_ident()))
            # print('Recognition results: {0}'.format(pred_str[0]))
            # must add the fullpath because same numbered crop can belong to multiple images:
            # image1/crop0.jpg, image1/crop1.jpg, image2/crop0.jpg, image2/crop1.jpg etc

            # the file that contains the recognition result for each cropped image

            rec_results[crop_name] = {'result': rec_result, 'stn_input_with_fid': stn_input_with_fid,
                                'stn_output': stn_output}
        return rec_results

    except Exception as e:
        if logger:
            logger.exception(e)
        raise