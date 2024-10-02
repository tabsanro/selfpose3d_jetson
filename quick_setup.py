import os
import re
import sys
import shutil
import zipfile

import gdown
import requests
from tqdm import tqdm

lib_path = os.path.join(os.path.dirname(__file__), 'modules', 'SelfPose3d', 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from modules import SelfPose3d

def main():
    # SelfPose3D repository -- by tabsanro
    SelfPose3d.download_from_drive.run()

if __name__ in '__main__':
    main()
