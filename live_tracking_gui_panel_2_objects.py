"""Browser control panel for TWO-object live tracking, served by viser.

The two-object twin of live_tracking_gui_panel.py, for
live_tracking_new_gui_2_objects.py. The same design rule holds: the tracking
loop never pays for the UI. No camera frames are rendered here; the viser
callbacks only flip flags and stash values, the loop drains them once per
iteration with a dict copy under a lock, and a low-rate thread pushes the
status text back into the browser.

What is per object: re-register (from its prompt or by clicking in a popup),
the text prompt, and the mesh. What is
shared: recording, saving, discarding and stopping (one take holds both pose
streams, frame for frame).
"""
from __future__ import annotations

import copy
import threading


def _empty_requests():
    return {'reregister': [False, False], 'reregister_click': [False, False],
            'prompt': [None, None],
            'mesh': [None, None], 'save': False, 'discard': False,
            'quit': False, 'auto_record': None}


class GuiState2:
    """Shared state between the viser thread and the tracking loop.

    objects: two dicts with keys name, text_prompt, mesh_path, mask_mode,
    upright (the loop keeps them current with publish_object)."""

    def __init__(self, objects, auto_record=True):
        self._lock = threading.Lock()
        # loop -> gui
        self.fps = 0.0
        self.recording = False
        self.n_buffered = 0
        self.last_event = 'starting up'
        self.takes_saved = 0
        self.auto_record = auto_record
        self.objects = [dict(o) for o in objects]
        # gui -> loop (one-shot requests)
        self._req = _empty_requests()

    # ---- called from the viser thread --------------------------------
    def request(self, key, value=True):
        """A shared request: save, discard, quit, auto_record."""
        with self._lock:
            self._req[key] = value

    def request_object(self, key, obj, value=True):
        """A per-object request: reregister, prompt, mesh. obj is 0 or 1."""
        with self._lock:
            self._req[key][obj] = value

    # ---- called from the tracking loop -------------------------------
    def take_requests(self):
        """Return pending requests and reset them. Cheap: one small copy."""
        with self._lock:
            out = self._req
            self._req = _empty_requests()
            return out

    def publish(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def publish_object(self, obj, **kw):
        with self._lock:
            self.objects[obj].update(kw)

    def snapshot(self):
        with self._lock:
            return {
                'fps': self.fps, 'recording': self.recording,
                'n_buffered': self.n_buffered, 'last_event': self.last_event,
                'takes_saved': self.takes_saved,
                'auto_record': self.auto_record,
                'objects': copy.deepcopy(self.objects),
            }


def start_gui_2(state: GuiState2, port: int, mesh_choices=None):
    """Build the panel and return (server, stop_fn). Non-blocking.

    mesh_choices: mesh paths offered in both objects' dropdowns."""
    import viser

    server = viser.ViserServer(port=port,
                               label='FoundationPose live tracking: 2 objects',
                               verbose=False)

    gui = server.gui
    status = gui.add_markdown('starting…')

    with gui.add_folder('Tracking'):
        btn_both = gui.add_button('Re-register both')
        btn_both.on_click(lambda _: (state.request_object('reregister', 0),
                                     state.request_object('reregister', 1)))
        btn_1 = gui.add_button('Re-register object 1')
        btn_1.on_click(lambda _: state.request_object('reregister', 0))
        btn_2 = gui.add_button('Re-register object 2')
        btn_2.on_click(lambda _: state.request_object('reregister', 1))
        # These open the box / points popup (point_prompt_mask.py) on the
        # workstation's screen, whatever the object's mask mode.
        btn_c1 = gui.add_button('Re-register object 1 by clicking (popup)')
        btn_c1.on_click(lambda _: state.request_object('reregister_click', 0))
        btn_c2 = gui.add_button('Re-register object 2 by clicking (popup)')
        btn_c2.on_click(lambda _: state.request_object('reregister_click', 1))

    with gui.add_folder('Recording'):
        chk_auto = gui.add_checkbox('Auto-record (policy-gated)',
                                    initial_value=state.auto_record)
        chk_auto.on_update(lambda _: state.request('auto_record',
                                                   chk_auto.value))
        btn_save = gui.add_button('Save take now')
        btn_save.on_click(lambda _: state.request('save'))
        btn_discard = gui.add_button('Discard buffer')
        btn_discard.on_click(lambda _: state.request('discard'))

    def _object_folder(k):
        o = state.objects[k]
        with gui.add_folder(f'Object {k + 1} (swap mid-run)'):
            txt_prompt = gui.add_text(f'Text prompt {k + 1}',
                                      initial_value=o['text_prompt'] or '')
            btn_prompt = gui.add_button(f'Apply prompt {k + 1} + re-register')
            btn_prompt.on_click(
                lambda _: state.request_object('prompt', k, txt_prompt.value))
            if mesh_choices:
                dd_mesh = gui.add_dropdown(
                    f'Mesh {k + 1}', options=tuple(mesh_choices),
                    initial_value=(o['mesh_path'] if o['mesh_path'] in mesh_choices
                                   else mesh_choices[0]))
                btn_mesh = gui.add_button(f'Load mesh {k + 1} + re-register')
                btn_mesh.on_click(
                    lambda _: state.request_object('mesh', k, dd_mesh.value))
            txt_mesh = gui.add_text(f'…or mesh path {k + 1}',
                                    initial_value=o['mesh_path'])
            btn_mesh2 = gui.add_button(f'Load path {k + 1} + re-register')
            btn_mesh2.on_click(
                lambda _: state.request_object('mesh', k, txt_mesh.value))

    _object_folder(0)
    _object_folder(1)

    with gui.add_folder('Session'):
        btn_quit = gui.add_button('Stop and save')
        btn_quit.on_click(lambda _: state.request('quit'))

    stop = threading.Event()

    def _object_table(k, o):
        # One narrow table per object: the panel is a sidebar.
        return (f"**object {k + 1}**\n\n| | |\n|---|---|\n"
                f"| name | `{o['name']}` |\n"
                f"| prompt | `{o['text_prompt'] or '—'}` |\n"
                f"| mesh | `{(o['mesh_path'] or '—').split('/')[-1]}` |\n"
                f"| mask | `{o['mask_mode']}` |\n"
                f"| topic | `{o['topic']}` |\n"
                f"| upright | {'yes' if o['upright'] else 'no'} |\n\n")

    def _pump():
        # 4 Hz is plenty for a status readout and keeps the GUI entirely off
        # the tracking loop's thread.
        while not stop.is_set():
            s = state.snapshot()
            rec = '🔴 RECORDING' if s['recording'] else '⚪ idle'
            status.content = (
                f"### {rec}\n\n"
                f"| | |\n|---|---|\n"
                f"| loop | **{s['fps']:.1f}** FPS (both objects) |\n"
                f"| buffered | {s['n_buffered']} frames |\n"
                f"| takes saved | {s['takes_saved']} |\n\n"
                + _object_table(0, s['objects'][0])
                + _object_table(1, s['objects'][1])
                + f"_{s['last_event']}_"
            )
            stop.wait(0.25)

    threading.Thread(target=_pump, name='gui_pump', daemon=True).start()
    return server, stop.set
