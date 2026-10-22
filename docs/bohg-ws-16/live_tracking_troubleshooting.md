# live_tracking_new.py — flaky errors that vanish on retry

Diagnosed 2026-09-15 on `bohg-ws-16`.

Three failure modes look like real bugs but are not. They come and go across
*identical* commands, which is the tell. This note explains each one so the next
person (or the next you) doesn't spend an hour chasing a phantom.

Command under investigation:

```bash
python third_party/FoundationPose/live_tracking_new.py \
  --depth_backend zed_neural --camera_resolution HD1080 --camera_fps 30 --fps 40 \
  --mesh_path assets/urdf/dextoolbench/eraser/handle_eraser/handle_eraser.obj \
  --save_dir videos/Sept_15_debug/ --ros
```

---

## 1. `UnboundLocalError` inside Python's regex compiler

```
UnboundLocalError: local variable 'skip' referenced before assignment    sre_compile.py:208
UnboundLocalError: local variable 'group' referenced before assignment   sre_parse.py:847
UnboundLocalError: local variable 'lo' referenced before assignment      sre_compile.py:197
```

**Cause: mashing Ctrl-C.** These are artifacts of the interrupt, not bugs.

All three are inside Python's own regex compiler, which cannot fail this way
normally. The obvious suspect — a corrupt bytecode cache — was ruled out:

```
sre_compile.pyc OK 15147 bytes   (dated Nov 2022; no truncated .pyc anywhere in the env)
```

What actually happens: CPython raises `KeyboardInterrupt` at an arbitrary
*bytecode boundary*. When that lands inside deeply recursive stdlib code, the
frame's locals aren't all bound yet at the moment the traceback is built, so you
get a nonsense error naming whichever local hadn't been assigned. Which variable
it names (`skip` / `group` / `lo`) is just where the signal happened to land.

It hits the regex compiler specifically because
`torch/utils/hipify/hipify_python.py:729` compiles one enormous trie pattern at
import time — slow and deeply recursive (note `[Previous line repeated 3 more
times]` in the traceback), so it's a wide target for a stray signal.

The same cause produces the scattered `KeyboardInterrupt`s in `linecache`,
`tokenize`, and `importlib._bootstrap_external`.

**Why retrying "fixes" it:** the outcome depends on *when* the Ctrl-C lands, not
on the code. Nothing was broken, so nothing needed fixing.

**Fix:** press Ctrl-C once and wait. The mashing is the cause. Startup is slow
(see §4), so it feels unresponsive — resist the urge.

---

## 2. `multiprocessing/spawn.py` traceback — a real latent bug

```
File "<string>", line 1, in <module>
  ... spawn_main → _fixup_main_from_path → runpy.run_path ...
File ".../live_tracking_new.py", line 37, in <module>
    import nvdiffrast.torch as dr
```

The `if __name__ == '__main__'` guard at line 533 only protects `main()`. With
the `spawn` start method, the child runs `runpy.run_path`, which **re-executes
the entire module top level** — including line 37's `import nvdiffrast.torch`.

Symptoms that confirm it:

- `Warp 1.0.2 initialized` printed **twice**
- `[buffer] retaining newest 2400 frames in RAM` printed twice
- three ZED `Logging level INFO` lines

Every capture child pays a full torch/nvdiffrast/Warp import. That is slow *and*
it widens the window in which a stray Ctrl-C triggers §1.

**Fix:** move the heavy top-level imports inside `main()` (or make them lazy).

---

## 3. `Segmentation fault (core dumped)`

Camera contention. A previous run's capture child still held `/dev/video0` when
the next run tried to open the ZED.

Check before relaunching:

```bash
lsof /dev/video*
ps -eo pid,stat,etime,cmd | grep live_tracking
```

A process in `D` state (uninterruptible) will not die on Ctrl-C — wait for it to
clear. Note the parent may sit in `D` while its capture child works normally;
that is not necessarily a hang.

---

## 4. Background: why the box feels slow

Not CPU, RAM, GPU, or disk — all near idle (96.5% idle CPU on 32 cores, 2.4 GB
of 62 GB used, no swap, 6% disk). The tell is **iowait on an idle box**.

`$HOME` is AFS and the Python interpreter itself lives there:

```
HOME=/afs/cs.stanford.edu/u/kedia
/proc/<pid>/exe -> /afs/cs.stanford.edu/u/kedia/.local/share/uv/python/cpython-3.11.13.../python3.11
```

The repo and conda envs are on NFS (`/juno`). So shell startup, `vim`,
tab-completion, and every Python import are network round-trips. Processes park
in `folio_wait_bit_common` waiting on filesystem pages.

This is why startup feels hung, which is what prompts the Ctrl-C mashing in §1.

Also worth knowing: AFS tokens expire (`tokens` to check). Once they lapse,
anything AFS-backed hangs hard rather than erroring.

Longer-term fix: relocate the uv Python and caches off AFS onto local `/`
(1.7 TB free, NVMe).

---

## Quick triage

| Symptom | Real bug? | Action |
|---|---|---|
| `UnboundLocalError` in `sre_*` | No | Stop mashing Ctrl-C; rerun cleanly |
| `KeyboardInterrupt` in `linecache`/`tokenize` | No | Same |
| `spawn.py` → `runpy.run_path` traceback | Yes | Move heavy imports into `main()` |
| `Segmentation fault` | No | Wait for the stale process to release `/dev/video0` |
| `ModuleNotFoundError: tensorrt` | Yes | Env issue — use `.venv_isaacsim` (`drl`), not `fp` |

Rule of thumb: if an identical command succeeds on retry, suspect signal timing
or a stale process — not your code.
