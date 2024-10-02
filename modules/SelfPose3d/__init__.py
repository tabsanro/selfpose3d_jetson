import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
add_path(this_dir)

lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

# print(sys.path)   # test

from tools import download_from_drive
from lib.models.model_loader import load_model
from lib.core.config import config, update_config