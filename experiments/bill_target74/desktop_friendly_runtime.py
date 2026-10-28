"""Windows scheduling wrapper; keep registered model implementations unchanged."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = Path(__file__).with_name("desktop_friendly_policy.json")
POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
OUTPUT = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/desktop_friendly"
_LOG: Path | None = None
_WRITE_LOCK = threading.Lock()


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                *[(name, ctypes.c_ulonglong) for name in (
                    "total_phys", "available_phys", "total_page", "available_page",
                    "total_virtual", "available_virtual", "available_extended")]]


def _kernel():
    if os.name != "nt":
        raise RuntimeError("desktop-friendly scheduling requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.GetPriorityClass.restype = wintypes.DWORD
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                              ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.POINTER(ctypes.c_size_t)]
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    return kernel


def memory_snapshot() -> dict:
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not _kernel().GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"available_memory_gib": memory.available_phys / 2**30,
            "total_memory_gib": memory.total_phys / 2**30,
            "memory_load_percent": int(memory.load)}


def native_settings() -> dict:
    kernel = _kernel()
    process = kernel.GetCurrentProcess()
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"priority_class": int(kernel.GetPriorityClass(process)),
            "affinity_mask": int(allowed.value),
            "logical_processors": int(allowed.value.bit_count()),
            "pid": os.getpid()}


def emit(event: str, **detail) -> None:
    if _LOG is None:
        raise RuntimeError("run this module through the desktop-friendly runtime wrapper")
    row = {"event": event, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "process_cpu_seconds": time.process_time(), **detail}
    with _WRITE_LOCK, _LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def wait_before_fit(context: str) -> dict:
    """Wait without consuming a fit start; never terminate an active model fit."""
    if os.environ.get("BILL_DESKTOP_FRIENDLY_ACTIVE") != "1" or _LOG is None:
        raise RuntimeError("a fit requires the desktop-friendly runtime wrapper")
    required_memory = (
        POLICY.get("minimum_available_memory_before_module_gib", POLICY["minimum_available_memory_gib"])
        if context == "before_module_import" else POLICY["minimum_available_memory_gib"]
    )
    start = time.monotonic()
    last_report = -60.0
    while True:
        snapshot = memory_snapshot()
        elapsed = time.monotonic() - start
        if snapshot["available_memory_gib"] >= required_memory:
            emit("memory_gate_passed", context=context, waited_seconds=elapsed,
                 required_available_memory_gib=required_memory, **snapshot)
            return snapshot
        if elapsed - last_report >= 60:
            emit("waiting_for_memory", context=context, waited_seconds=elapsed,
                 required_available_memory_gib=required_memory, **snapshot)
            last_report = elapsed
        time.sleep(POLICY["memory_poll_seconds"])


def _acquire_campaign_lock(lock, msvcrt, *, wait_for_memory: bool) -> None:
    lock.seek(0, os.SEEK_END)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()

    required_memory = POLICY.get(
        "minimum_available_memory_before_module_gib",
        POLICY["minimum_available_memory_gib"],
    )
    while True:
        if wait_for_memory:
            wait_before_fit("before_module_import")

        lock_wait_start = time.monotonic()
        last_lock_report = -60.0
        acquired = False
        while not acquired:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as exc:
                if exc.errno not in {13, 36}:
                    raise
                waited = time.monotonic() - lock_wait_start
                if waited - last_lock_report >= 60:
                    emit("waiting_for_campaign_lock", waited_seconds=waited)
                    last_lock_report = waited
                time.sleep(10)
                if wait_for_memory:
                    snapshot = memory_snapshot()
                    if snapshot["available_memory_gib"] < required_memory:
                        emit("yielding_campaign_lock_wait_for_memory",
                             waited_seconds=waited,
                             required_available_memory_gib=required_memory,
                             **snapshot)
                        break

        if not acquired:
            continue
        if not wait_for_memory:
            return

        snapshot = memory_snapshot()
        if snapshot["available_memory_gib"] >= required_memory:
            emit("memory_gate_passed", context="before_module_import_after_campaign_lock",
                 required_available_memory_gib=required_memory, **snapshot)
            return

        emit("released_campaign_lock_for_memory_wait",
             required_available_memory_gib=required_memory, **snapshot)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _configure() -> dict:
    kernel = _kernel()
    process = kernel.GetCurrentProcess()
    before = native_settings()
    bits = [bit for bit in range(before["affinity_mask"].bit_length())
            if before["affinity_mask"] & (1 << bit)]
    selected = bits[-POLICY["max_logical_processors"]:]
    mask = sum(1 << bit for bit in selected)
    if not kernel.SetPriorityClass(process, 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.SetProcessAffinityMask(process, mask):
        raise ctypes.WinError(ctypes.get_last_error())
    os.environ.update(POLICY["environment"])
    os.environ["BILL_DESKTOP_FRIENDLY_ACTIVE"] = "1"
    after = native_settings()
    if after["priority_class"] != 0x4000 or after["affinity_mask"] != mask:
        raise RuntimeError("Windows did not apply the requested scheduling settings")
    return {"before": before, "after": after,
            "environment": {key: os.environ[key] for key in POLICY["environment"]}}


def main() -> None:
    import msvcrt

    parser = argparse.ArgumentParser()
    parser.add_argument("--module")
    parser.add_argument("--probe", action="store_true")
    args, remaining = parser.parse_known_args()
    if not args.probe and not (args.module or "").startswith("experiments.bill_target74."):
        parser.error("--module must name an experiments.bill_target74 module")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Export this running instance so experiment imports share its log and policy state.
    sys.modules["experiments.bill_target74.desktop_friendly_runtime"] = sys.modules[__name__]
    global _LOG
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _LOG = OUTPUT / f"runtime_{stamp}_{os.getpid()}.jsonl"
    with (OUTPUT / "campaign.lock").open("a+b") as lock:
        settings = _configure()
        _acquire_campaign_lock(lock, msvcrt, wait_for_memory=not args.probe)
        emit("runtime_start", module=args.module, python_executable=sys.executable,
             policy=POLICY, settings=settings,
             **memory_snapshot())
        print(json.dumps({"runtime_log": str(_LOG), "settings": settings["after"]}), flush=True)
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(POLICY["resource_sample_seconds"]):
                emit("resource_sample", settings=native_settings(), **memory_snapshot())

        sampler = threading.Thread(target=sample, name="resource-sampler", daemon=True)
        sampler.start()
        try:
            if args.probe:
                print(json.dumps({"policy": POLICY, "settings": settings,
                                  "memory": memory_snapshot()}, indent=2), flush=True)
            else:
                if args.module != "experiments.bill_target74.desktop_git_scan":
                    from experiments.bill_target74 import runner as campaign

                    append = campaign._append

                    def gated_append(event: dict) -> None:
                        if event.get("event") == "fit_start":
                            wait_before_fit(f"fit_start:{event.get('card')}:{event.get('month')}")
                        append(event)

                    campaign._append = gated_append
                sys.argv = [args.module, *remaining]
                runpy.run_module(args.module, run_name="__main__")
        except BaseException as exc:
            emit("runtime_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            emit("runtime_complete", **memory_snapshot())
        finally:
            stop.set()
            sampler.join(timeout=2)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
