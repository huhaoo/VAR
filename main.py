################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

var_ckpt='var16_finetune_1500.pth'
# var_ckpt='var16_finetune.pth'

# build vae, var
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

############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
# class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
class_labels = [0,1]*4  #@param {type:"raw"}
# print(class_labels)
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# sample
B = len(class_labels)
all_images = []

def dft(img):
    from torch.fft import fft2,fftshift
    # print(torch.mean(img))
    img=fft2(img)
    img=fftshift(img)
    img=torch.log(torch.abs(img) + 1)
    # print(torch.mean(img))
    return img
    min_val = img[i].min()
    max_val = img[i].max()
    return (img - min_val) / (max_val - min_val + 1e-6)


lst_chw=None
for i in reversed(range(len(patch_nums))):
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth,dropout_layer=i)

    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    if lst_chw is None:
        lst_chw = torch.zeros_like(chw)
    chw,lst_chw=torch.abs(chw-lst_chw),chw
    chw = chw.permute(1, 2, 0)

    # print(chw.shape, chw.min(), chw.max())
    W=recon_B3HW.shape[3]
    if True:
     if False:
        for i in range(chw.shape[1]//W):
            # chw[:, i*W:(i+1)*W, :] = (0.299 * chw[:, i*W:(i+1)*W, 0] + 0.587 * chw[:, i*W:(i+1)*W, 1] + 0.114 * chw[:, i*W:(i+1)*W, 2])[:, :, None]
            for j in range(3):
                chw[:, i*W:(i+1)*W, j] = dft(chw[:, i*W:(i+1)*W, j])
            MX=12
            assert MX*0.5<chw[:, i*W:(i+1)*W, :].max()<MX
            chw[:, i*W:(i+1)*W, :] = chw[:, i*W:(i+1)*W, :] / MX

    chw = chw.cpu().numpy()
    chw = PImage.fromarray((chw*255).astype(np.uint8))
    # chw.show()
    all_images.append(chw)
all_images.append(PImage.fromarray((lst_chw.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)))



total_height = sum(img.height for img in all_images)
max_width = max(img.width for img in all_images)  # 所有图片宽度应相同

# 创建空白画布
combined = PImage.new('RGB', (max_width, total_height))

# 垂直拼接所有图像
y_offset = 0
for img in all_images:
    combined.paste(img, (0, y_offset))
    y_offset += img.height

# 保存最终合并图像
combined.save('figure.png')