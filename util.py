from torch.fft import fft2,fftshift
import torch
import math
import numpy
import random

def init(seed=0):
	# seed
	torch.manual_seed(seed)
	random.seed(seed)
	numpy.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# run faster
	tf32 = True
	torch.backends.cudnn.allow_tf32 = bool(tf32)
	torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
	torch.set_float32_matmul_precision('high' if tf32 else 'highest')

def dft(img):
    img=fft2(img)
    img=fftshift(img)
    img=torch.log(torch.abs(img) + 1)
    return img