"""Export FoundationPose's RefineNet to ONNX for TRT engine building.

Wraps RefineNet so it returns a (trans, rot) tuple instead of a dict
(ONNX can't export dict outputs cleanly). Batch axis is dynamic.

Usage:
    python third_party/FoundationPose/scripts/export_refine_onnx.py \
        --out third_party/FoundationPose/weights/2023-10-28-18-33-37/refine_net.onnx
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# FoundationPose's nested package layout uses sys.path appends; mirror them.
_FP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FP_DIR))
sys.path.insert(0, str(_FP_DIR / 'learning' / 'models'))
sys.path.insert(0, str(_FP_DIR / 'learning' / 'training'))

from refine_network import RefineNet  # noqa: E402


import torch.nn.functional as F  # noqa: E402


def _force_legacy_transformer_path(module: nn.Module):
    """Both nn.TransformerEncoderLayer and nn.MultiheadAttention have
    fast-path ops in PyTorch eval mode (`aten::_transformer_encoder_layer_fwd`
    and `aten::_native_multi_head_attention`) that lack ONNX symbolics.
    Monkey-patch each instance's `forward` to a manual implementation
    that traces to standard ONNX ops (matmul / softmax / linear)."""
    for m in module.modules():
        if isinstance(m, nn.TransformerEncoderLayer):
            def _slow_te_fwd(src, src_mask=None, src_key_padding_mask=None,
                             is_causal=False, _layer=m):
                x = src
                x = _layer.norm1(
                    x + _layer._sa_block(
                        x, src_mask, src_key_padding_mask, is_causal=False))
                x = _layer.norm2(x + _layer._ff_block(x))
                return x
            m.forward = _slow_te_fwd
        elif isinstance(m, nn.MultiheadAttention):
            if not m._qkv_same_embed_dim:
                raise NotImplementedError(
                    "patched MHA only supports self-attention with same "
                    "embed dim (the FP RefineNet / ScoreNet case).")

            def _slow_mha_fwd(query, key, value,
                              key_padding_mask=None, need_weights=False,
                              attn_mask=None, average_attn_weights=True,
                              is_causal=False, _mha=m):
                # (B, L, D)
                B, L, D = query.shape
                H = _mha.num_heads
                head_dim = D // H
                qkv = F.linear(query, _mha.in_proj_weight, _mha.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.reshape(B, L, H, head_dim).transpose(1, 2)
                k = k.reshape(B, L, H, head_dim).transpose(1, 2)
                v = v.reshape(B, L, H, head_dim).transpose(1, 2)
                attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn = F.softmax(attn, dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).reshape(B, L, D)
                out = _mha.out_proj(out)
                return out, None
            m.forward = _slow_mha_fwd


class RefineNetExportWrapper(nn.Module):
    """Returns (trans, rot) instead of a dict so ONNX export can name them.

    Also bypasses RefineNet.forward's `bs = len(A); cat([A,B]); x[:bs]; x[bs:]`
    pattern, which traces the slice indices as Python int constants and
    forces TRT to fix the batch dim to 1. We run `encodeA` separately on
    A and B (same weights, twice the GPU launches at bs=1 but no shape-
    dependent slicing), then continue with the original pipeline.

    The reshape after `encodeAB` uses `-1` for the batch dim and hard-codes
    the spatial dim to 400 (20x20 = H/8 x W/8 at H=W=160) so the resulting
    Reshape input is constant w.r.t. batch but symbolic-batch-friendly.
    """

    def __init__(self, model: RefineNet):
        super().__init__()
        self.model = model
        # Spatial size after encodeA(stride=4) then encodeAB(stride=2) -> H/8.
        # H=W=160 -> 20x20 = 400 tokens.
        self._n_tokens = 400

    def forward(self, A, B):
        m = self.model
        a = m.encodeA(A)
        b = m.encodeA(B)
        ab = torch.cat((a, b), 1).contiguous()
        ab = m.encodeAB(ab)
        ab = m.pos_embed(
            ab.reshape(-1, ab.shape[1], self._n_tokens).permute(0, 2, 1))
        trans = m.trans_head(ab).mean(dim=1)
        rot = m.rot_head(ab).mean(dim=1)
        return trans, rot


def _patch_cfg_defaults(cfg):
    """Mirror PoseRefinePredictor.__init__'s missing-key fallbacks."""
    defaults = {
        'use_normal': False, 'use_mask': False, 'use_BN': False,
        'c_in': 4, 'crop_ratio': 1.2, 'n_view': 1,
        'trans_rep': 'tracknet', 'rot_rep': 'axis_angle', 'zfar': 3,
        'normalize_xyz': False, 'normal_uint8': False,
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default='2023-10-28-18-33-37',
                   help='Weight folder under third_party/FoundationPose/weights/')
    p.add_argument('--weight', default='model_best.pth')
    p.add_argument('--out', default=None,
                   help='Output .onnx path. Default: <weights_dir>/refine_net.onnx')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--validate_bs', type=int, default=1,
                   help='Run a PyTorch vs ONNX-Runtime sanity check at this batch size.')
    p.add_argument('--no_validate', action='store_true',
                   help='Skip the ONNX-Runtime numerical check at the end.')
    args = p.parse_args()

    weights_dir = _FP_DIR / 'weights' / args.run_name
    ckpt_path = weights_dir / args.weight
    cfg_path = weights_dir / 'config.yml'
    if args.out is None:
        args.out = str(weights_dir / 'refine_net.onnx')

    cfg = OmegaConf.load(str(cfg_path))
    cfg['ckpt_dir'] = str(ckpt_path)
    cfg['enable_amp'] = True
    _patch_cfg_defaults(cfg)

    print(f'[export] loading {ckpt_path}')
    model = RefineNet(cfg=cfg, c_in=cfg['c_in']).cuda()
    ckpt = torch.load(str(ckpt_path))
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model.cuda().eval()

    wrapper = RefineNetExportWrapper(model).cuda().eval()
    _force_legacy_transformer_path(wrapper)

    H = W = int(cfg['input_resize'][0])
    c_in = int(cfg['c_in'])
    A = torch.randn(1, c_in, H, W, dtype=torch.float32, device='cuda')
    B = torch.randn(1, c_in, H, W, dtype=torch.float32, device='cuda')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f'[export] writing ONNX (c_in={c_in}, H=W={H}, opset={args.opset}) -> {args.out}')
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (A, B), args.out,
            opset_version=args.opset,
            input_names=['A', 'B'],
            output_names=['trans', 'rot'],
            dynamic_axes={
                'A':     {0: 'batch'},
                'B':     {0: 'batch'},
                'trans': {0: 'batch'},
                'rot':   {0: 'batch'},
            },
        )
    print('[export] ONNX export done')

    if args.no_validate:
        return

    # Numerical sanity check: torch fp32 vs ONNX-Runtime CPU fp32.
    try:
        import onnxruntime as ort
    except ImportError:
        print('[export] onnxruntime not installed — skipping numerical check')
        return

    bs = args.validate_bs
    A_v = torch.randn(bs, c_in, H, W, dtype=torch.float32, device='cuda')
    B_v = torch.randn(bs, c_in, H, W, dtype=torch.float32, device='cuda')
    with torch.inference_mode():
        torch_trans, torch_rot = wrapper(A_v, B_v)
    torch_trans = torch_trans.cpu().numpy()
    torch_rot = torch_rot.cpu().numpy()

    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    ort_outs = sess.run(['trans', 'rot'], {
        'A': A_v.cpu().numpy(), 'B': B_v.cpu().numpy(),
    })
    ort_trans, ort_rot = ort_outs

    def _maxabs(a, b):
        return float(np.max(np.abs(a - b)))

    print(f'[export] validate bs={bs}: '
          f'trans maxabs={_maxabs(torch_trans, ort_trans):.4e}, '
          f'rot maxabs={_maxabs(torch_rot, ort_rot):.4e}')


if __name__ == '__main__':
    main()
