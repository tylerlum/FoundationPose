# Live tracking runbook (bohg-ws-16)

How to run FoundationPose live tracking on bohg-ws-16, which command is known
to work, and how to A/B the settings while chasing the open post-save
slowdown.

Written for whoever (human or agent) picks this up next. **Claims below are
labelled VERIFIED (measured on this machine, date given) or UNVERIFIED.**
Please keep that distinction when editing.

---

## 0. Environment

```bash
ssh tylerlum@bohg-ws-16
drl                 # activates the venv, sets caches/toolchain/ROS, cds to the repo
```

`drl` is a shell function from `/home/tylerlum/.drl_env.sh`, sourced by
`~/.bashrc`. Everything lives on the local NVMe at
`/home/tylerlum/github_repos/depthbasedRL`. See `docs/LOCAL_SETUP.md` for how
the environment was built and why.

Sanity check the env at any time:

```bash
python /home/tylerlum/verify_env.py      # from the repo root
```

---

## 1. Which script to run

| script | what it adds | state |
|---|---|---|
| `live_tracking_new.py` | the original | untouched, works |
| `live_tracking_new_autorecord.py` | record only while the RL policy publishes | works |
| `live_tracking_new_gui.py` | + viser browser panel, swap object/prompt mid-run | **the one in use**; has an open bug, see §4 |

All live in `third_party/FoundationPose/`.

---

## 2. THE SAFE PATH

**VERIFIED 2026-09-19**: this exact command ran 140 s with four saves. Each
save blocked the loop 0.6–0.7 s and the loop held ~30 FPS throughout, with no
degradation after any save. HD1080 caps the ZED at 30 FPS, so ~30 is the
ceiling here, not a shortfall.

```bash
drl
python third_party/FoundationPose/live_tracking_new_gui.py \
  --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path assets/urdf/dextoolbench/spatula/spoon_spatula/spoon_spatula.obj \
  --save_dir /home/tylerlum/_test \
  --mask_mode center_box --no_auto_record --ros --no_robot_frame
```

What makes this the safe path, versus the configuration that showed the bug:

- `--save_dir` on local disk (though `/juno` was later ruled out as a cause)
- `--mask_mode center_box` (no SAM3 resident on the GPU, no clicking)
- `--no_auto_record` (no `/iiwa/joint_cmd` subscriber)
- `--no_robot_frame` (publishes only `camera_frame/current_object_pose`)
- run against a **throwaway local roscore**, not the bohg-ws-2 master
- **no browser attached** to the viser panel
- `DISPLAY` unset, so no OpenCV window

To use a local roscore instead of bohg-ws-2:

```bash
$DRL_VENV/bin/rosmaster --core -p 11397 &
export ROS_MASTER_URI=http://localhost:11397 ROS_IP=127.0.0.1
```

After each save the script prints `[autorecord] save blocked the loop for X.Xs`
and the loop prints `[fps] achieved inst=.. avg40=.. n=..`. Those two lines are
the whole diagnostic: watch whether `avg40` returns to its pre-save value.

---

## 3. Options worth flipping

### Masking

| flag | effect |
|---|---|
| `--mask_mode center_box` | fixed centre box, no clicking, no model. Fastest, needs the object centred. |
| `--text_prompt "black spoon"` | SAM3 text→mask, no clicking. **VERIFIED**: score 0.97, 0.22 s per call once warm. Loads facebook/sam3 (840M params) onto the GPU and keeps it there. |
| `--mask_mode sam` (default) | the original interactive 4-corner picker. Needs a display. |
| `--text_backend gdino` | Grounding DINO + SAM instead of SAM3. Ungated fallback. |
| `--text_fallback_click` | if a text prompt matches nothing, fall back to clicking. |

Prompt wording matters a lot. **VERIFIED** on the lab scene: `"black spoon"`
→ 0.97, `"spoon"` → 0.95, `"ladle"` → **nothing at all**, same object.

`facebook/sam3` is a gated HF repo. The token at
`/home/tylerlum/.cache/huggingface/token` has access. If you regenerate it,
copy it there — `$HOME` is AFS and `HF_HOME` points at local disk.

### Recording

| flag | effect |
|---|---|
| (default) | auto-record: buffer only while `--policy_topic` publishes |
| `--no_auto_record` | buffer continuously, save on demand / at exit |
| `--policy_topic` | default `/iiwa/joint_cmd`; `rl_policy_node.py` publishes it and `/sharpa/joint_cmd` together at 60 Hz |
| `--policy_idle_timeout` | default 0.5 s = 30 missed messages at 60 Hz |

**VERIFIED**: against a simulated policy with two bursts (5.0 s, 4.0 s), this
produced exactly two takes of 5.43 s and 3.84 s. Idle time excluded. The delta
is the idle timeout.

### Saving / encoding

| flag | default | effect |
|---|---|---|
| `--encode_preset` | `veryfast` | x264 preset |
| `--encode_crf` | `20` | rate control; higher = smaller and faster |
| `--no_overlay_video` | off | skip `overlay.mp4`, roughly halves encode |
| `--viz_data` | **off** | write `viz_data.npz` (40–90 MB/take) |
| `--viz_compress` | off | compress it (3.83 s vs 0.09 s to write) |

**VERIFIED 2026-09-19** on 1200 frames decoded from a real take
(`20260919_211555`), `save_run()` timed end to end:

| config | time | size |
|---|---|---|
| defaults, written to `/juno` | 2.4 s | 13.9 MB |
| defaults, written to local disk | 2.0 s | 13.9 MB |
| `--no_overlay_video`, local | 1.0 s | 6.9 MB |
| `ultrafast` + crf 26, no overlay | 0.7 s | 6.6 MB |
| `ultrafast` + crf 30, no overlay | 0.6 s | 3.8 MB |

History, for context: the original settings passed no x264 preset (so ffmpeg
used `medium`) and used imageio's `quality=8`, which maps to a qscale **libx264
does not honour** — that is why early takes were 140–190 MB. Old vs new on 400
synthetic frames: 8.71 s / 146 MB → 1.43 s / 10.3 MB.

There is **no NVENC** in the bundled `imageio-ffmpeg` binary, so hardware
encoding is not available without installing a fuller ffmpeg.

### GUI

| flag | effect |
|---|---|
| (default) | viser panel on an auto-picked free port; URL printed at startup |
| `--gui_port N` | pin the port |
| `--no_gui` | no panel; falls back to the OpenCV keyboard window |
| `--keyboard_window` | force the OpenCV window on *alongside* the panel |
| `--mesh_choices a.obj b.obj` | extra meshes in the swap dropdown |
| `--selftest_swap` | drive the panel from a timer (headless testing) |
| `--selftest_saves` | fire repeated saves on a timer |

**VERIFIED**: mid-run swaps via the panel — prompt change 0.81 s, mesh swap
0.90 s (vs ~25 s for a full restart). Driven through `--selftest_swap`, i.e.
the same code path the browser callbacks use; **the widgets themselves have
not been clicked by a test**.

---

## 4. RESOLVED: loop dropped to ~1.6 Hz after the first save

**Fixed 2026-09-19, commit `8933d88`. Kept here because the cause is
non-obvious and will bite anyone who adds another ROS subscriber.**

### Cause

Not the save, not SAM3, not the browser, not the save directory. It was the
`/iiwa/joint_cmd` subscriber that `PolicyActivityWatcher` creates.

When a ROS publisher **exits**, rospy is left holding a half-closed socket.
`recv()` then returns `b''` immediately rather than blocking, forever, and it
logs a full traceback every iteration. Profiled with py-spy on a reproduced
stall:

```
45.4%  recv_buff   (tcpros_base.py:104)
41.0%  findCaller  (roslogging.py:64/66/67)
stack: robust_connect_subscriber -> receive_loop -> receive_once -> recv_buff
```

It is a pure-Python loop, so it **holds the GIL** and starves the whole
process: one thread pinned at ~103% of a core, the main thread blocked on a
futex, 30 FPS -> 1.6 FPS, and the viser panel unresponsive.

It looked like "after the first save" because the policy stopping is what
*both* closes the take and strands the socket. The save was a bystander.

### Fix

The subscription runs in a child process (`_policy_watch_proc`); the main
side reads a timestamp out of shared memory. The spin, if it happens, burns
a core over there where it cannot touch the tracking loop's GIL.

### Things that do NOT work (measured, do not retry)

- Recycling the `Subscriber` when the publisher goes quiet — still spins.
- `unregister()` then resubscribing when the master lists a publisher — the
  master keeps **stale** registrations, so it flip-flops and respins.
- **`unregister()` does not stop rospy's receive thread.** This is the fact
  that forces process isolation.

### Simpler designs, if this ever needs revisiting

1. Drop ROS auto-record; record from the GUI button. Removes the failure
   mode instead of containing it.
2. Poll the master for a publisher instead of subscribing (~10 lines, no
   subprocess) — but that detects "policy node running", not "publishing",
   so it only works if the node starts and exits per rollout.

### How it was found

`py-spy record -- python <tracker>` (py-spy must be the PARENT;
`ptrace_scope=1` forbids attaching to a sibling), then rank threads by
**non-idle** CPU. Ranking by total time just surfaces threads parked in
`select`/`wait`. See `scripts/local_setup/analyze_prof2.py`.

The reproduction is `scripts/local_setup/run_full.sh`: full config against a
**local** roscore, with `fake_policy_pub.py` standing in for the real policy.
It never touches bohg-ws-2, so it cannot move the robot.

---

## 5. Gotchas

- **Camera contention.** A stale capture child can survive a hard Ctrl-C and
  hold `/dev/video0`; the next run then dies with `CAMERA NOT DETECTED`.
  Check `fuser -v /dev/video0`, clear with
  `bash /home/tylerlum/cleanup_tracking.sh`.
  That script matches on `python` command lines containing `live_tracking_new`,
  `rosmaster`, `fake_policy_pub` — **an `ssh` command line containing those
  strings matches too**, so it can kill its own invocation. Keep it in its own
  command.
- **Import order.** `rospy` must be imported *before* the FoundationPose stack
  or it segfaults reliably. The scripts handle this; new scripts must too.
- **Groups.** `/dev/video0` needs `video`, `/usr/local/zed` needs `zed`. Both
  granted; group changes need a fresh login.
- **Ctrl-C once, then wait.** Mashing it produces bogus `UnboundLocalError`s
  from inside `sre_compile` — see `docs/live_tracking_troubleshooting.md` §1.

---

## 6. Test harnesses

In `scripts/local_setup/`:

| file | purpose |
|---|---|
| `verify_env.py` | import + version check, ends with `ALL OK` |
| `test_zed.py` | open the camera, grab 30 frames, report intrinsics/timing |
| `test_e2e.py` | headless capture → register → track, no SAM |
| `test_reregister.py` | the `r` path headlessly (SAM + re-register timing) |
| `test_sam3_text.py` | SAM3 text prompting on a live frame, writes overlays |
| `fake_policy_pub.py` | stand-in for `rl_policy_node.py`; publishes JointState bursts |
| `cleanup_tracking.sh` | kill stale trackers, free the camera |

All except `verify_env.py` need the camera.
