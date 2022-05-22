import torch
from model.eval_network import RDE_VOS

model = torch.load(r"pretrain\model_s012_best_yv.pth")
print(1)
prop_model = RDE_VOS().cuda().eval()
prop_model.load_state_dict(model)