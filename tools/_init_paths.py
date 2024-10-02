import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

modules_path = osp.join(this_dir, '..', 'modules')
add_path(modules_path)

configs_path = osp.join(this_dir, '..', 'configs')
add_path(configs_path)

# # import SelfPose3d 하면 다음 에러가 발생한다.
# # ModuleNotFoundError: No module named 'utils'
# # 이는 sys path에 lib가 추가되어 있지 않아서 발생하는 문제이다.
# # 즉, lib path를 추가해주어야 한다.
# lib_path = osp.join(this_dir, '..', 'modules', 'SelfPose3d', 'lib')
# add_path(lib_path)

# SelfP

# 다행히 focus와 mivolo는 본인 root인 "focus.", "mivolo."로 시작하므로 추가하지 않아도 된다.
