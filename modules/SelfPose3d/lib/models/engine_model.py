import contextlib
from collections import namedtuple
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt

class MyLogger(trt.ILogger):
    def __init__(self, level=trt.Logger.INFO):
        trt.ILogger.__init__(self)
        self.level = level.value
    
    def log(self, severity, msg):
        if severity.value <= self.level:
            print(f'[{severity.name:^14}] {msg}')

TRT_LOGGER = MyLogger(trt.Logger.INFO)

def check_version(library_name, min, max):
    try:
        import sys
        if sys.version_info >= (3, 8):
            import importlib.metadata as metadata
            version = metadata.version(library_name)
        else:
            import pkg_resources as metadata
            version = metadta.get_distribution(library_name).version
        
        TRT_LOGGER.log(trt.Logger.INFO, f'{library_name} version: {version}')
        from packaging import version as pkg_version
        if min <= pkg_version.parse(version) <= max:
            TRT_LOGGER.log(trt.Logger.INFO, f'{library_name} version is compatible')
        else:
            TRT_LOGGER.log(trt.Logger.ERROR, f'{library_name} version is not compatible')
    except (metadata.PackageNotFoundError, metadata.DistributionNotFound):
        TRT_LOGGER.log(trt.Logger.ERROR, f'{library_name} is not installed')

def defalut_class_names(data):
    # if data:
    #     with contextlib.suppress(Exception):
    #         return yaml_load(check_yaml(data))['names']
    return {i: f'class{i}' for i in range(999)}

def check_class_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f'{n}-class dataset requires class indices 0-{n-1}, but you have invalid class indices '
                f'{min(names.keys())}-{max(names.keys())} defined in your dataset YAML.'
            )
        # if isinstance(names[0], str) and names[0].startswith('n0'):
        #     # imagenet class i.e. 'n01440764
        #     names_map = yaml_load(ROOT / 'cfg/datasets/ImageNet.yaml')['map']
        #     # human readable names
        #     names = {k: names_map[v] for k, v in names.items()}
        return names

class EngineModel(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        weights: str,
        device=torch.device("cuda:0"),
        data=None,
        fp16=False,
        batch=1,
        fuse=False,
        verbose=False,
        copy=False,
    ):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, nn.Module)
        engine = weights.endswith(".engine")
        fp16 &= engine
        model, metadata = None, None

        # Set device
        cuda = torch.cuda.is_available() and device.type != 'cpu'
        if cuda and not any([nn_module, engine]):
            device = torch.device('cpu')
            cuda = False

        if nn_module:
            # TODO
            pass
        elif engine:
            try:
                import tensorrt as trt
            except ImportError:
                import platform
                if platform.system() == 'Linux':
                    check_version('tensorrt', '7.0.0', '10.1.0')
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            with open(w, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())

            try:
                context = model.create_execution_context()
            except Exception as e:
                TRT_LOGGER.log(trt.Logger.ERROR, f'Error: {e}')
                raise e

            bindings = OrderedDict()
            output_names = []
            fp16 = False
            dynamic = False
            is_trt10 = not hasattr(model, 'num_bindings')
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                            if dtype == np.float16:
                                fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                            if dtype == np.float16:
                                fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['input'].shape[0] # YOLO는 images
        else:
            raise TypeError(
                f'model={w} is not a supported model format.'
            )

        # Check names
        if 'names' not in locals():
            names = defalut_class_names(data)
        names = check_class_names(names)

        # Update __dict__
        self.__dict__.update(locals())

    def forward(self, im):
        if isinstance(im, np.ndarray):
            im = torch.from_numpy(im).to(self.device)
        b, ch, h, w = im.shape
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()

        if self.nn_module:
            pass
        elif self.engine:
            if self.dynamic or im.shape != self.bindings['input'].shape:
                if self.is_trt10:
                    self.context.set_input_shape('input', im.shape)
                    self.bindings['input'] = self.bindings['input']._replace(shape=im.shape)
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    i = self.model.get_binding_index('input')
                    self.context.set_binding_shape(i, im.shape)
                    self.bindings['input'] = self.bindings['input']._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

            s = self.bindings['input'].shape
            assert im.shape == s, f'input size {im.shape} {">" if self.dynamic else "not equal to"} max model size {s}'
            self.binding_addrs['input'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        else:
            pass

        # y = [x if isinstance(x, np.ndarray) else x.cpu().numpy() for x in y]
        # if isinstance(y, (list, tuple)):
        #     # if len(self.names) == 999 and (self.task == 'segment' or len(y) == 2):
        #     if len(self.names) == 999 and len(y) == 2:
        #         # segment and names not defined
        #         ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)
        #         nc = y[ib].shape[1] - y[ip].shape[3] - 4
        #         self.names = {i: f'class{i}' for i in range(nc)}
        #     return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        if self.batch_size == 1:
            return self.from_numpy(self.copy_tensor(y[0]) if self.copy else y[0])
        return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warm_up(self, imgsz=(1, 3, 512, 960)):
        # TODO
        pass

    def copy_tensor(self, y):
        _y = torch.empty_like(y)
        _y.copy_(y)
        return _y