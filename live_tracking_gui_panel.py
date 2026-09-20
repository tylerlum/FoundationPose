"""Browser control panel for live tracking, served by viser.

Design rule: the tracking loop must never pay for the UI. No camera frames
are rendered here, and no GUI call happens inside the loop's hot path. The
viser callbacks only flip flags and stash values; the loop drains them once
per iteration with a dict copy under a lock, which is nanoseconds. A separate
low-rate thread pushes status text back into the browser.

This replaces the OpenCV 'Keyboard Control' window, which cost real
throughput: the loop had to call cv2.imshow + cv2.waitKey(1) EVERY iteration
(skipping it makes waitKey stall for hundreds of ms on some HighGUI
backends). Measured on this rig, dropping that window took the loop from
36 to 40 FPS at a 40 FPS cap, and from ~60 to ~75 FPS uncapped.
"""
from __future__ import annotations

import threading
import time


class GuiState:
    """Shared state between the viser thread and the tracking loop."""

    def __init__(self, text_prompt=None, mesh_path='', auto_record=True,
                 mask_mode='sam'):
        self._lock = threading.Lock()
        # loop -> gui
        self.fps = 0.0
        self.recording = False
        self.n_buffered = 0
        self.object_name = ''
        self.mesh_path = mesh_path
        self.text_prompt = text_prompt or ''
        self.mask_mode = mask_mode
        self.last_event = 'starting up'
        self.takes_saved = 0
        # gui -> loop (one-shot requests)
        self._req = {'reregister': False, 'save': False, 'discard': False,
                     'quit': False, 'prompt': None, 'mesh': None,
                     'auto_record': None}
        self.auto_record = auto_record

    # ---- called from the viser thread --------------------------------
    def request(self, key, value=True):
        with self._lock:
            self._req[key] = value

    # ---- called from the tracking loop -------------------------------
    def take_requests(self):
        """Return pending requests and reset them. Cheap: one dict copy."""
        with self._lock:
            out = dict(self._req)
            self._req = {'reregister': False, 'save': False, 'discard': False,
                         'quit': False, 'prompt': None, 'mesh': None,
                         'auto_record': None}
            return out

    def publish(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self):
        with self._lock:
            return {
                'fps': self.fps, 'recording': self.recording,
                'n_buffered': self.n_buffered, 'object_name': self.object_name,
                'mesh_path': self.mesh_path, 'text_prompt': self.text_prompt,
                'mask_mode': self.mask_mode, 'last_event': self.last_event,
                'takes_saved': self.takes_saved,
                'auto_record': self.auto_record,
            }


def start_gui(state: GuiState, port: int, mesh_choices=None):
    """Build the panel and return (server, stop_fn). Non-blocking."""
    import viser

    server = viser.ViserServer(port=port, label='FoundationPose live tracking',
                               verbose=False)

    gui = server.gui
    status = gui.add_markdown('starting…')

    with gui.add_folder('Tracking'):
        btn_rereg = gui.add_button('Re-register', icon=None)
        btn_rereg.on_click(lambda _: state.request('reregister'))

    with gui.add_folder('Recording'):
        chk_auto = gui.add_checkbox('Auto-record (policy-gated)',
                                    initial_value=state.auto_record)
        chk_auto.on_update(lambda _: state.request('auto_record',
                                                   chk_auto.value))
        btn_save = gui.add_button('Save take now')
        btn_save.on_click(lambda _: state.request('save'))
        btn_discard = gui.add_button('Discard buffer')
        btn_discard.on_click(lambda _: state.request('discard'))

    with gui.add_folder('Object (swap mid-run)'):
        txt_prompt = gui.add_text('Text prompt', initial_value=state.text_prompt)
        btn_prompt = gui.add_button('Apply prompt + re-register')
        btn_prompt.on_click(lambda _: state.request('prompt', txt_prompt.value))

        if mesh_choices:
            dd_mesh = gui.add_dropdown('Mesh', options=tuple(mesh_choices),
                                       initial_value=mesh_choices[0])
            btn_mesh = gui.add_button('Load mesh + re-register')
            btn_mesh.on_click(lambda _: state.request('mesh', dd_mesh.value))
        txt_mesh = gui.add_text('…or mesh path',
                                initial_value=state.mesh_path)
        btn_mesh2 = gui.add_button('Load path + re-register')
        btn_mesh2.on_click(lambda _: state.request('mesh', txt_mesh.value))

    with gui.add_folder('Session'):
        btn_quit = gui.add_button('Stop and save')
        btn_quit.on_click(lambda _: state.request('quit'))

    stop = threading.Event()

    def _pump():
        # 4 Hz is plenty for a status readout and keeps the GUI entirely off
        # the tracking loop's thread.
        while not stop.is_set():
            s = state.snapshot()
            rec = '🔴 RECORDING' if s['recording'] else '⚪ idle'
            status.content = (
                f"### {rec}\n\n"
                f"| | |\n|---|---|\n"
                f"| loop | **{s['fps']:.1f}** FPS |\n"
                f"| buffered | {s['n_buffered']} frames |\n"
                f"| takes saved | {s['takes_saved']} |\n"
                f"| object | `{s['object_name']}` |\n"
                f"| mask mode | `{s['mask_mode']}` |\n"
                f"| prompt | `{s['text_prompt'] or '—'}` |\n"
                f"| mesh | `{s['mesh_path'].split('/')[-1]}` |\n\n"
                f"_{s['last_event']}_"
            )
            stop.wait(0.25)

    threading.Thread(target=_pump, name='gui_pump', daemon=True).start()
    return server, stop.set
