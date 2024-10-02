try:
    import torch
    print(torch.__version__)
except ImportError:
    raise ValueError('CUDA is not available. Please check your configuration.')


if torch.cuda.is_available():
    print('CUDA is available.')
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))
else:
    raise ValueError('CUDA is not available. Please check your configuration.')

