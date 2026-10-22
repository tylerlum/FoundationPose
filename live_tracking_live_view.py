"""Live views in the tracker's viser panel ('Live view' folder), each switched on and off from the panel:
  3D: the ZED point cloud and both meshes at their tracked poses, in the robot frame;
  2D: the camera image with both meshes (projected vertices) and their axes drawn on.
The 3D view also shows the camera image inside the camera's frustum ('image in frustum'), with the 2D view's pose
annotations unless 'frustum: annotated' is off.

With both off the tracking loop pays one bool check per frame. On, the loop hands over references to the frame it
already has (no copies; fetch() returns fresh arrays every frame) and a background thread draws and sends at the
panel's rate, so the loop never waits on viser.
"""
import math
import threading

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_COLORS = ((0, 200, 0), (220, 0, 220))   # object 1 / 2, as in the mask popups
_AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
_MAX_2D_VERTS = 3000
_FRUSTUM_IMG_W = 240       # the image in the frustum is this wide (240x135 for 16:9)


def _wxyz(R):
    return tuple(float(v) for v in Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]])


class LiveView3D:
    def __init__(self, server, K, T_RC, wh, view_3d=False, view_2d=False, hz=5.0, stride=4, table_z=0.53,
                 look_dist=0.5):
        self.server, self.K, self.T = server, np.asarray(K, float), np.asarray(T_RC, float)
        self.W, self.H = wh
        self._slot = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self._mesh_handles = [None, None]
        self._mesh_ids = [None, None]
        self._verts_2d = [None, None]       # (id(mesh), subsampled vertices) for the 2D view

        sc, gui = server.scene, server.gui
        sc.set_up_direction('+z')
        self._static = [sc.add_frame('/live/robot_base', axes_length=0.15, axes_radius=0.004),
                        sc.add_grid('/live/table', width=2.0, height=2.0, position=(0.0, 0.0, table_z),
                                    cell_size=0.1)]
        c, R = self.T[:3, 3], self.T[:3, :3]
        # The camera (OpenCV axes) with a thin frustum, and the image on the frustum's far plane (z = d), sized from K
        # as in simtoolreal_dev's replay_bundle.py.
        # Viser normalises the frustum by its volume, so scale = d * cbrt(tan^2(fov/2) * aspect / 3) puts its far
        # plane at z = d.
        d, fov, aspect = 0.15, 2 * math.atan(self.H / 2 / self.K[1, 1]), self.W / self.H
        self._static.append(sc.add_frame('/live/camera', position=tuple(float(v) for v in c), wxyz=_wxyz(R),
                                         axes_length=0.05, axes_radius=0.002))
        sc.add_camera_frustum('/live/camera/frustum', fov=fov, aspect=aspect,
                              scale=d * (math.tan(fov / 2) ** 2 * aspect / 3) ** (1 / 3), thickness=0.002,
                              color=(255, 255, 0))
        self._fimg = sc.add_image('/live/camera/image', np.zeros((9, 16, 3), np.uint8),
                                  render_width=d * self.W / self.K[0, 0], render_height=d * self.H / self.K[1, 1],
                                  position=(0.0, 0.0, d), format='jpeg', jpeg_quality=70, visible=False,
                                  cast_shadow=False, receive_shadow=False)
        self._cloud = sc.add_point_cloud('/live/cloud', points=np.zeros((1, 3), np.float32),
                                         colors=np.zeros((1, 3), np.uint8), point_size=0.003)

        # The browser starts at x = y = 0, the camera's height, upright, looking where the ZED looks.
        target = c + look_dist * R[:, 2]
        server.initial_camera.position = (0.0, 0.0, float(c[2]))
        server.initial_camera.look_at = tuple(float(v) for v in target)
        server.initial_camera.up = (0.0, 0.0, 1.0)
        server.initial_camera.fov = 2 * math.atan(self.H / 2 / self.K[1, 1])

        with gui.add_folder('Live view (off = no cost)'):
            self.chk = gui.add_checkbox('3D view', initial_value=view_3d)
            self.chk_2d = gui.add_checkbox('2D view', initial_value=view_2d)
            self.chk_cloud = gui.add_checkbox('point cloud', initial_value=True)
            self.chk_mesh = gui.add_checkbox('meshes', initial_value=True)
            self.chk_fimg = gui.add_checkbox('image in frustum', initial_value=True)
            self.chk_fimg_ann = gui.add_checkbox('frustum: annotated', initial_value=True)
            self.sl_hz = gui.add_slider('update Hz', min=1.0, max=30.0, step=1.0, initial_value=hz)
            self.sl_stride = gui.add_slider('cloud stride', min=1, max=16, step=1, initial_value=stride)
            self.sl_psize = gui.add_slider('point size', min=0.001, max=0.02, step=0.0005, initial_value=0.003)
            self.sl_opacity = gui.add_slider('mesh opacity', min=0.1, max=1.0, step=0.05, initial_value=0.8)
            self.chk_nodepth = gui.add_checkbox('2D: red = no depth', initial_value=True)
            self.sl_scale_2d = gui.add_slider('2D image scale', min=0.25, max=1.0, step=0.25, initial_value=0.5)
            self.img = gui.add_image(np.zeros((self.H // 2, self.W // 2, 3), np.uint8), label='camera + poses',
                                     format='jpeg', jpeg_quality=75)
        self.sl_psize.on_update(lambda _: setattr(self._cloud, 'point_size', self.sl_psize.value))
        self.sl_opacity.on_update(lambda _: self._set_opacity())
        for h in (self.chk, self.chk_2d, self.chk_cloud, self.chk_mesh, self.chk_fimg):
            h.on_update(lambda _: self._set_visible())
        self._set_visible()

        vv, uu = np.mgrid[0:self.H, 0:self.W]
        self._uu, self._vv = uu.astype(np.float32), vv.astype(np.float32)
        threading.Thread(target=self._run, name='live_view_3d', daemon=True).start()

    # ---- called from the tracking loop -----------------------------------------------------------------------
    def offer(self, rgb, depth, poses, meshes):
        """Hand the latest frame to the view thread. A no-op while the view is off."""
        if not self.enabled:
            return
        with self._lock:
            self._slot = (rgb, depth, [None if p is None else np.array(p) for p in poses], list(meshes))
        self._new.set()

    def stop(self):
        self._stop.set()
        self._new.set()

    # ---- the view thread --------------------------------------------------------------------------------------
    def _set_visible(self):
        self.on_3d, self.on_2d = bool(self.chk.value), bool(self.chk_2d.value)
        self.enabled = self.on_3d or self.on_2d
        for h in self._static:
            h.visible = self.on_3d
        self._cloud.visible = self.on_3d and self.chk_cloud.value
        self._fimg.visible = self.on_3d and self.chk_fimg.value
        for h in self._mesh_handles:
            if h is not None:
                h.visible = self.on_3d and self.chk_mesh.value
        self.img.visible = self.on_2d

    def _set_opacity(self):
        for h in self._mesh_handles:
            if h is not None:
                h.opacity = self.sl_opacity.value

    def _run(self):
        while not self._stop.is_set():
            self._new.wait()
            self._new.clear()
            if self._stop.is_set():
                break
            with self._lock:
                slot, self._slot = self._slot, None
            if slot is None or not self.enabled:
                continue
            try:
                self._draw(*slot)
            except Exception as e:   # never take the tracker down for a view
                print(f'\n[live_view] {e!r}', flush=True)
            self._stop.wait(1.0 / max(self.sl_hz.value, 0.1))

    def _draw(self, rgb, depth, poses, meshes):
        fimg = self.on_3d and self.chk_fimg.value
        ann = None
        if self.on_2d or (fimg and self.chk_fimg_ann.value):   # one annotated image serves both
            ann = self._annotate(rgb, depth, poses, meshes)
        if self.on_2d:
            self.img.image = ann
        if fimg:
            src = ann if (ann is not None and self.chk_fimg_ann.value) else rgb
            self._fimg.image = cv2.resize(src, (_FRUSTUM_IMG_W, round(_FRUSTUM_IMG_W * self.H / self.W)),
                                          interpolation=cv2.INTER_AREA)
        if self.on_3d:
            self._draw_3d(rgb, depth, poses, meshes)

    def _annotate(self, rgb, depth, poses, meshes):
        """The 2D view: the image at the '2D image scale' with both meshes and their axes drawn on."""
        sc = float(self.sl_scale_2d.value)
        img = rgb if sc == 1.0 else cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img).copy()
        if self.chk_nodepth.value:
            # the depth FoundationPose gets (before its own erode / bilateral filter): none here -> red
            bad = ~(np.isfinite(depth) & (depth >= 0.001))
            if sc != 1.0:
                bad = cv2.resize(bad.astype(np.uint8), (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            img[bad] = (0.5 * img[bad] + (127, 0, 0)).astype(np.uint8)
        K = self.K.copy()
        K[:2] *= sc
        for k, (pose, mesh) in enumerate(zip(poses, meshes)):
            if mesh is None or pose is None or not np.all(np.isfinite(pose)):
                continue
            if self._verts_2d[k] is None or self._verts_2d[k][0] != id(mesh):
                v = np.asarray(mesh.vertices, np.float64)
                if len(v) > _MAX_2D_VERTS:
                    v = v[np.random.default_rng(0).choice(len(v), _MAX_2D_VERTS, replace=False)]
                self._verts_2d[k] = (id(mesh), v)
            v = self._verts_2d[k][1]
            q = v @ pose[:3, :3].T + pose[:3, 3]
            q = q[q[:, 2] > 1e-3]
            uv = (q[:, :2] / q[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
            uv = np.round(uv).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
            img[uv[ok, 1], uv[ok, 0]] = _COLORS[k]
            # the object's axes, 5 cm
            P = np.vstack([np.zeros(3), 0.05 * np.eye(3)]) @ pose[:3, :3].T + pose[:3, 3]
            if np.all(P[:, 2] > 1e-3):
                p = (P[:, :2] / P[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
                o = tuple(int(x) for x in np.round(p[0]))
                for e, c in zip(p[1:], _AXIS_COLORS):
                    cv2.line(img, o, tuple(int(x) for x in np.round(e)), c, 2, cv2.LINE_AA)
                cv2.putText(img, f'obj{k + 1}', (o[0] + 5, o[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            _COLORS[k], 1, cv2.LINE_AA)
        return img

    def _draw_3d(self, rgb, depth, poses, meshes):
        if self.chk_cloud.value:
            s = int(self.sl_stride.value)
            d = depth[::s, ::s]
            ok = np.isfinite(d) & (d > 0.001)
            z = d[ok]
            u, v = self._uu[::s, ::s][ok], self._vv[::s, ::s][ok]
            K = self.K
            pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
            pts = pts @ self.T[:3, :3].T + self.T[:3, 3]
            self._cloud.points = pts.astype(np.float32)
            self._cloud.colors = rgb[::s, ::s][ok]
        if self.chk_mesh.value:
            for k, (pose, mesh) in enumerate(zip(poses, meshes)):
                if mesh is None:
                    continue
                if self._mesh_ids[k] != id(mesh):     # first frame, or the mesh was swapped
                    if self._mesh_handles[k] is not None:
                        self._mesh_handles[k].remove()
                    self._mesh_handles[k] = self.server.scene.add_mesh_simple(
                        f'/live/object_{k + 1}', vertices=np.asarray(mesh.vertices, np.float32),
                        faces=np.asarray(mesh.faces, np.uint32), color=_COLORS[k],
                        opacity=self.sl_opacity.value)
                    self._mesh_ids[k] = id(mesh)
                h = self._mesh_handles[k]
                if pose is None or not np.all(np.isfinite(pose)):
                    h.visible = False
                    continue
                T = self.T @ pose
                h.position = tuple(float(x) for x in T[:3, 3])
                h.wxyz = _wxyz(T[:3, :3])
                h.visible = self.on_3d
