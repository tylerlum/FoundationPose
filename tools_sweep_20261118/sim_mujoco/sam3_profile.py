import sys, time, numpy as np, torch, cv2
sys.path.insert(0, '/home/tylerlum/github_repos/depthbasedRL/third_party/FoundationPose')
import robust_tool
from text_mask import mask_from_text
fr = [cv2.cvtColor(cv2.imread(f'takes/occlusion_tour/replay/rgb/{i:05d}.png'), cv2.COLOR_BGR2RGB) for i in range(60)]
m0 = cv2.imread('takes/occlusion_tour/replay/masks_1/00000.png', 0) > 0
def tm(f, n=10):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
print(f'P SAM3 text detection ("red brush", warm): {tm(lambda: mask_from_text(fr[0], "red brush"), 5):.0f} ms', flush=True)
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
dev = 'cuda'
model = Sam3TrackerVideoModel.from_pretrained('facebook/sam3').to(dev, dtype=torch.bfloat16).eval()
proc = Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
print('P params (M): total', round(sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
for size in (1008, 784, 644):
    try:
        sess = proc.init_video_session(inference_device=dev, dtype=torch.bfloat16)
        tp, tmod = [], []
        for i, f in enumerate(fr):
            torch.cuda.synchronize(); a = time.time()
            inp = proc(images=f, device=dev, return_tensors='pt', size={'height': size, 'width': size})
            torch.cuda.synchronize(); b = time.time()
            if i == 0:
                proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=1,
                    input_masks=m0.astype(np.float32), original_size=inp.original_sizes[0])
            with torch.inference_mode():
                o = model(inference_session=sess, frame=inp.pixel_values[0].to(torch.bfloat16))
            torch.cuda.synchronize(); c = time.time()
            if i > 5: tp.append(b - a); tmod.append(c - b)
        m = proc.post_process_masks([o.pred_masks], original_sizes=inp.original_sizes, binarize=True)[0]
        print(f'P SAM3 tracker at {size}px: preprocess {np.median(tp)*1000:.1f} ms + model {np.median(tmod)*1000:.1f} ms; '
              f'pixel_values {tuple(inp.pixel_values.shape)}, last mask px {int(m.sum())}', flush=True)
    except Exception as e:
        print(f'P SAM3 tracker at {size}px failed: {type(e).__name__}: {str(e)[:150]}', flush=True)
