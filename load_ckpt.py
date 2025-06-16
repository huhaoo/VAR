import torch

ckpt_path = 'local_output/ar-ckpt-best.pth' 
save_path = 'var16_finetune.pth'                     

checkpoint = torch.load(ckpt_path, map_location='cpu')

trainer_state = checkpoint.get('trainer', {})
var_state_dict = trainer_state.get('var_wo_ddp', None)

if var_state_dict is None:
    raise KeyError("var_wo_ddp not found")

torch.save(var_state_dict, save_path)
print(f"[✓] Saved to {save_path}")
