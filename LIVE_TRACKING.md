# Live ZED tracking

Live 6-DoF object pose tracking with a ZED camera, with optional ROS
publishing, text-prompted masking, policy-gated recording, and a browser
control panel.

Originally developed inside `kushal2000/depthbasedRL` under
`third_party/FoundationPose/`; moved here so it lives with FoundationPose.

## Scripts

| script | what it is |
|---|---|
| `live_tracking_new.py` | the base entry point |
| `live_tracking_new_autorecord.py` | + record only while an RL policy publishes |
| `live_tracking_new_gui.py` | + viser browser panel, swap object/prompt mid-run |

Each is a superset of the one above it, so you can adopt them one at a time.

## Run

```bash
python live_tracking_new_gui.py \
  --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path /path/to/object.obj \
  --save_dir /path/to/output \
  --text_prompt "black spoon" --ros
```

The viser panel URL is printed at startup (auto-picked free port). It gives
re-register, save, discard, an auto-record toggle, live loop rate, and
mid-run object/prompt swapping.

Without `--text_prompt` you get the original interactive 4-corner SAM
picker. `--mask_mode center_box` skips masking models entirely.

## Masking

| flag | effect |
|---|---|
| `--text_prompt "..."` | SAM3 text -> mask. No clicking, ever. ~0.22s/call warm. |
| `--text_backend gdino` | Grounding DINO + SAM instead. Ungated. |
| `--text_fallback_click` | fall back to the picker if the prompt matches nothing |
| `--mask_mode center_box` | fixed centre box, no model |

`facebook/sam3` is a **gated** HF repo: accept the licence and have a token.
Prompt wording matters more than you would expect — on one scene
`"black spoon"` scored 0.97, `"spoon"` 0.95, and `"ladle"` matched *nothing*,
for the same object.

Masking runs at registration only, never per frame.

## Recording

Default in the autorecord/gui scripts: buffer only while `--policy_topic`
(default `/iiwa/joint_cmd`) is being published, and write the take when it
stops. Tracking itself never pauses, so starting a new rollout costs no
re-register.

`--no_auto_record` buffers continuously and saves on demand instead.

## Saving

Defaults: x264 `veryfast` at `-crf 20`, `viz_data.npz` off. Measured on 1200
real frames:

| config | time | size |
|---|---|---|
| old defaults (no preset, `quality=8`) | 8.7s | 146 MB |
| `veryfast` + `crf 20` | 2.4s | 13.9 MB |
| + `--no_overlay_video` | 1.0s | 6.9 MB |

`quality=8` maps to a qscale libx264 does **not** honour, which is why takes
used to come out at 140-190 MB. Saves block the tracking loop deliberately
and print how long they took.

## Requirements beyond stock FoundationPose

- `pyzed` from the ZED SDK (not the PyPI package of that name)
- `rospy` etc. for `--ros`. `rospypi/simple` ships ROS 1 Noetic wheels for
  modern Python, so no apt or conda is needed.
- `viser` for the GUI panel, `transformers` for SAM3 / Grounding DINO
- `tensorrt` + `cuda-python` for the TRT pose refiner, which
  `live_tracking_new.py` uses **by default** via `--fp_refiner_trt`.
  Pass `--fp_refiner_trt ""` to stay on the PyTorch refiner.
- `--depth_backend fast_fs` expects `Fast-FoundationStereo` as a *sibling
  directory*. The `zed_neural` path has no such dependency.

## Gotchas

- **Import `rospy` BEFORE the FoundationPose stack.** Reversing it segfaults
  reliably. The scripts handle this; anything new must too.
- **A ROS subscriber whose publisher exits spins at 100% CPU** holding the
  GIL — `recv()` returns empty immediately instead of blocking, forever,
  logging each time. It dragged the tracking loop from 30 FPS to 1.6 FPS.
  `unregister()` does **not** stop it. That is why the policy watcher runs
  in a child process.
- **Camera contention.** A stale capture child can survive a hard Ctrl-C and
  hold `/dev/video0`; the next run then fails with `CAMERA NOT DETECTED`.
  Check with `fuser -v /dev/video0`.
- **Ctrl-C once, then wait.** Mashing it produces bogus `UnboundLocalError`s
  from inside Python's own regex compiler — an artifact of where the signal
  lands, not a real bug.
- Mesh/save-dir defaults in the scripts still point at depthbasedRL-relative
  paths. They are CLI arguments, so pass your own.
