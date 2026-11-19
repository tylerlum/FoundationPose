"""Full FoundationStereo (ViT-L 23-51-11) on DIR/left.png + right.png (960x540) -> DIR/fs_disp_960x540.npy (+ vis)."""
import os, sys
FS_DIR = os.environ.get('FOUNDATION_STEREO_DIR', '/home/tylerlum/github_repos/FoundationStereo')
sys.path.insert(0, FS_DIR)
import numpy as np, torch, imageio, cv2
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import vis_disparity
from core.foundation_stereo import FoundationStereo
D = sys.argv[1]; iters = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ckpt = os.environ.get('FOUNDATION_STEREO_CKPT', f'{FS_DIR}/pretrained_models/pretrained_models/23-51-11/model_best_bp2.pth')
cfg = OmegaConf.load(f'{os.path.dirname(ckpt)}/cfg.yaml'); cfg.setdefault('vit_size', 'vitl'); cfg['valid_iters'] = iters
torch.autograd.set_grad_enabled(False)
model = FoundationStereo(cfg); model.load_state_dict(torch.load(ckpt, weights_only=False)['model']); model.cuda().eval()
a = imageio.imread(f'{D}/left.png')[..., :3]; b = imageio.imread(f'{D}/right.png')[..., :3]; H, W = a.shape[:2]
t0, t1 = [torch.as_tensor(x).cuda().float()[None].permute(0, 3, 1, 2) for x in (a, b)]
padder = InputPadder(t0.shape, divis_by=32, force_square=False); t0, t1 = padder.pad(t0, t1)
with torch.cuda.amp.autocast(True):
    disp = model.forward(t0, t1, iters=iters, test_mode=True)
disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)
xx = np.arange(W)[None].repeat(H, 0); disp[(xx - disp) < 0] = 0   # not visible in the right image
np.save(f'{D}/fs_disp_960x540.npy', disp.astype(np.float32))
imageio.imwrite(f'{D}/fs_vis.png', np.concatenate([a, vis_disparity(disp)], 1))
print('FS done; disp range', float(disp[disp > 0].min()), float(disp.max()), 'mem GB', torch.cuda.max_memory_allocated() / 1e9)
