
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from models import VQVAE, build_vae_var

class sample:
	def __init__(self):# download checkpoint
		hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
		MODEL_DEPTH = 16
		vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
		if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
		if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

		patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if 'vae' not in globals() or 'var' not in globals():
			vae, var = build_vae_var(
				V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
				device=device, patch_nums=patch_nums,
				num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
			)

		# load checkpoints
		vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
		var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
		vae.eval(), var.eval()
		for p in vae.parameters(): p.requires_grad_(False)
		for p in var.parameters(): p.requires_grad_(False)
		print(f'prepare finished.')
		
		self.vae = vae
		self.var = var
		self.MODLE_DEPTH = MODEL_DEPTH
		self.patch_nums = patch_nums
		self.device = device
	
	def sample(self,input,mode='final'):
		recon_B3HW = self.var.autoregressive_infer_cfg(B=1, label_B=input, cfg=4, top_k=900, top_p=0.95)
