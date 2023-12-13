from functools import partial
import torch.nn as nn

from model.Inception_model import Inception
from model.GoogLeNet_model import GoogLeNet

"""==========Model config=========="""


MODEL_CONFIG = {
    "LOGNAME": "ours",
    "MODEL_TYPE": (Inception, GoogLeNet),
}

# MODEL_CONFIG = {
#     'LOGNAME': 'ours_small',
#     'MODEL_TYPE': (feature_extractor, flow_estimation),
#     'MODEL_ARCH': init_model_config(
#         F = 16,
#         W = 7,
#         depth = [2, 2, 2, 2, 2]
#     )
# }
