"""Mark an object by hand for SAM in a popup window: a box and/or points (13 Oct 2026).

For when a text prompt finds nothing. In the popup:

    drag            a box around the object (a new drag replaces it)
    left click      a point ON the object
    right click     a point that is NOT the object (also Ctrl + left click)

SAM redraws the mask after every change. Then:

    Enter / Space / s   accept            u   undo the last point
    m   SAM's next mask (whole / part / sub-part)       r   clear everything
    Esc / q   cancel (no mask)

A box alone usually does it for a tool. If the mask spills onto the hand or the
station, right-click there; if it misses part of the tool, left-click there.
For a small part (a screw or bolt head), drag a tight box.

    from point_prompt_mask import select_mask_interactive
    mask = select_mask_interactive(rgb, title='object 1: open_wrench_18')
    # rgb: H x W x 3 uint8, RGB. Returns an H x W uint8 0/1 mask, or None.

SAM is facebook/sam-vit-base, the model generate_mask.py's box picker loads.
The image is encoded once per popup; each change then costs only SAM's prompt
decoder (a few ms on the GPU). Needs an X display (the workstation's desktop,
or ssh -X).
"""
import contextlib
import os
import time

import cv2
import numpy as np
import torch

MODEL_ID = 'facebook/sam-vit-base'
_CACHE = {}


@contextlib.contextmanager
def _cpu_default_tensor():
    """FoundationPose's Utils.py makes CUDA the default tensor type during
    register / track_one; the HF processor and from_pretrained need the CPU
    default (as in live_tracking_new.py's _select_mask_safe)."""
    was_cuda = torch.tensor(0.).is_cuda
    if was_cuda:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        yield
    finally:
        if was_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


def _load(model_id):
    if model_id not in _CACHE:
        from transformers import SamModel, SamProcessor
        t0 = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with _cpu_default_tensor():
            model = SamModel.from_pretrained(model_id).to(device).eval()
            proc = SamProcessor.from_pretrained(model_id)
        _CACHE[model_id] = (model, proc, device)
        print(f'[points] loaded {model_id} on {device} in {time.time() - t0:.2f}s '
              '(cached for this process)', flush=True)
    return _CACHE[model_id]


class PromptMasker:
    """SAM masks for one image from a box and/or points (image coordinates)."""

    def __init__(self, rgb, model_id=MODEL_ID):
        from PIL import Image
        self.rgb = np.ascontiguousarray(rgb[..., :3])
        self.H, self.W = self.rgb.shape[:2]
        self.model, self.proc, self.device = _load(model_id)
        with _cpu_default_tensor():
            enc = self.proc(images=Image.fromarray(self.rgb), return_tensors='pt')
        self.original_sizes = enc['original_sizes']
        self.reshaped_input_sizes = enc['reshaped_input_sizes']
        new_h, new_w = (int(v) for v in self.reshaped_input_sizes[0])
        self._scale = np.array([new_w / self.W, new_h / self.H])   # image -> SAM input
        with torch.no_grad():
            self.embeddings = self.model.get_image_embeddings(
                enc['pixel_values'].to(self.device))
        self.reset()

    def reset(self):
        self.points, self.labels, self.box = [], [], None
        self.masks, self.scores, self.choice = None, None, None

    def add_point(self, x, y, positive=True):
        self.points.append([float(x), float(y)])
        self.labels.append(1 if positive else 0)
        self._predict()

    def set_box(self, x0, y0, x1, y1):
        self.box = [float(min(x0, x1)), float(min(y0, y1)),
                    float(max(x0, x1)), float(max(y0, y1))]
        self._predict()

    def undo(self):
        if self.points:
            self.points.pop()
            self.labels.pop()
        elif self.box is not None:
            self.box = None
        self._predict()

    def cycle(self):
        if self.masks is not None:
            self.choice = (self.choice + 1) % len(self.masks)

    @property
    def mask(self):
        if self.masks is None:
            return None
        return self.masks[self.choice]

    def _predict(self):
        if not self.points and self.box is None:
            self.masks = self.scores = self.choice = None
            return
        kw = {}
        if self.points:
            p = np.asarray(self.points) * self._scale
            kw['input_points'] = torch.tensor(p, dtype=torch.float32,
                                              device=self.device)[None, None]
            kw['input_labels'] = torch.tensor(self.labels, dtype=torch.long,
                                              device=self.device)[None, None]
        if self.box is not None:
            b = np.asarray(self.box).reshape(2, 2) * self._scale
            kw['input_boxes'] = torch.tensor(b.reshape(4), dtype=torch.float32,
                                             device=self.device)[None, None]
        with torch.no_grad():
            out = self.model(image_embeddings=self.embeddings,
                             multimask_output=True, **kw)
        masks = self.proc.post_process_masks(
            masks=out.pred_masks.cpu(), original_sizes=self.original_sizes,
            reshaped_input_sizes=self.reshaped_input_sizes)[0][0]   # (3, H, W)
        self.masks = [m.numpy().astype(bool) for m in masks]
        self.scores = out.iou_scores[0, 0].float().cpu().numpy()
        self.choice = int(np.argmax(self.scores))

    def render(self, color=(0, 200, 0), title='', drag=None):
        """BGR image: the frame, the mask, the box, the points and a key line."""
        img = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        m = self.mask
        if m is not None:
            tint = img.copy()
            tint[m] = color[::-1]
            img = cv2.addWeighted(tint, 0.45, img, 0.55, 0)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, color[::-1], 2)
        for b, c in ((self.box, (255, 255, 0)), (drag, (0, 255, 255))):
            if b is not None:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
        for (x, y), lab in zip(self.points, self.labels):
            cv2.circle(img, (int(x), int(y)), 7, (255, 255, 255), -1)
            cv2.circle(img, (int(x), int(y)), 5, (0, 200, 0) if lab else (0, 0, 255), -1)
        status = ('no mask yet: drag a box or click the object' if m is None else
                  f'mask {self.choice + 1}/3 (SAM score {self.scores[self.choice]:.2f}), '
                  f'{int(m.sum())} px')
        lines = [title,
                 'drag: box | left click: object | right click: not object',
                 'Enter: accept | u: undo | m: next mask | r: clear | Esc: cancel',
                 status]
        bar = 22 * len(lines) + 8
        img[:bar] = (0.35 * img[:bar]).astype(np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def select_mask_interactive(rgb, title='object', color=(0, 200, 0),
                            model_id=MODEL_ID):
    """Open the popup on `rgb` and return the accepted mask (H x W uint8 0/1),
    or None if cancelled or there is no display."""
    if not os.environ.get('DISPLAY'):
        print('[points] no DISPLAY: the popup needs the workstation desktop or '
              'ssh -X', flush=True)
        return None
    pm = PromptMasker(rgb, model_id)
    win = f'{title}  (drag a box / click, Enter to accept)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | getattr(cv2, 'WINDOW_KEEPRATIO', 0)
                    | getattr(cv2, 'WINDOW_GUI_NORMAL', 0))
    s = max(1.0, min(1600.0 / pm.W, 900.0 / pm.H))   # small frames get a bigger window
    cv2.resizeWindow(win, int(pm.W * s), int(pm.H * s))
    st = {'down': None, 'drag': None}

    def on_mouse(event, x, y, flags, _param):
        x = int(np.clip(x, 0, pm.W - 1))
        y = int(np.clip(y, 0, pm.H - 1))
        ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        if event == cv2.EVENT_LBUTTONDOWN:
            st['down'] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st['down'] is not None:
            x0, y0 = st['down']
            st['drag'] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif event == cv2.EVENT_LBUTTONUP and st['down'] is not None:
            x0, y0 = st['down']
            st['down'], st['drag'] = None, None
            if abs(x - x0) > 5 and abs(y - y0) > 5:
                pm.set_box(x0, y0, x, y)
            else:
                pm.add_point(x, y, positive=not ctrl)
        elif event == cv2.EVENT_RBUTTONDOWN:
            pm.add_point(x, y, positive=False)

    cv2.setMouseCallback(win, on_mouse)
    result = None
    while True:
        cv2.imshow(win, pm.render(color=color, title=title, drag=st['drag']))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32, ord('s')):
            if pm.mask is not None:
                result = pm.mask.astype(np.uint8)
                break
            print('[points] nothing marked yet', flush=True)
        elif key in (27, ord('q')):
            break
        elif key == ord('u'):
            pm.undo()
        elif key == ord('r'):
            pm.reset()
        elif key == ord('m'):
            pm.cycle()
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break                                  # closed with the window's X
        except cv2.error:
            break
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    print(f'[points] {title}: ' + (f'mask {int(result.sum())} px' if result is not None
                                   else 'cancelled'), flush=True)
    return result
