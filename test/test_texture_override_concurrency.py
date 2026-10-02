import inspect
import os
import threading
import time
import types

from pixal3d_extension.runtime import (
    PIXAL3D_TEXTURE_SIZE_ENV,
    _scoped_single_view_texture_export,
    _scoped_texture_size_env,
    _scoped_to_glb_texture_size,
    run_job,
)


def test_texture_override_serializes_environment_and_exporter(monkeypatch):
    sentinel = "preexisting-value"
    monkeypatch.setenv(PIXAL3D_TEXTURE_SIZE_ENV, sentinel)

    calls = []
    calls_lock = threading.Lock()

    def original_to_glb(*_args, **kwargs):
        with calls_lock:
            calls.append(
                (
                    threading.current_thread().name,
                    os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV),
                    kwargs.get("texture_size"),
                )
            )

    postprocess = types.SimpleNamespace(to_glb=original_to_glb)
    inference = types.SimpleNamespace(
        o_voxel=types.SimpleNamespace(postprocess=postprocess)
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()

    def worker(name, texture_size):
        if name == "second":
            second_attempting.set()
        with _scoped_texture_size_env(texture_size):
            with _scoped_to_glb_texture_size(texture_size, inference):
                if name == "first":
                    first_entered.set()
                    assert release_first.wait(2)
                else:
                    second_entered.set()
                postprocess.to_glb(texture_size=4096)

    first = threading.Thread(target=worker, args=("first", 1024), name="first")
    second = threading.Thread(target=worker, args=("second", 2048), name="second")
    first.start()
    assert first_entered.wait(2)
    second.start()
    assert second_attempting.wait(2)

    # The second request must not mutate the process environment while the
    # first request owns the exporter override.
    time.sleep(0.05)
    assert not second_entered.is_set()
    assert os.environ[PIXAL3D_TEXTURE_SIZE_ENV] == "1024"

    release_first.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive()
    assert not second.is_alive()
    assert calls == [
        ("first", "1024", 1024),
        ("second", "2048", 2048),
    ]
    assert os.environ[PIXAL3D_TEXTURE_SIZE_ENV] == sentinel
    assert postprocess.to_glb is original_to_glb


def test_base_texture_override_waits_for_multiview_run_lock(monkeypatch):
    from pixal3d_extension.multiview import _MV_RUN_LOCK

    sentinel = "preexisting-value"
    monkeypatch.setenv(PIXAL3D_TEXTURE_SIZE_ENV, sentinel)
    calls = []

    def original_to_glb(*_args, **kwargs):
        calls.append(
            (
                threading.current_thread().name,
                os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV),
                kwargs.get("texture_size"),
            )
        )

    postprocess = types.SimpleNamespace(to_glb=original_to_glb)
    inference = types.SimpleNamespace(
        o_voxel=types.SimpleNamespace(postprocess=postprocess)
    )
    multiview_entered = threading.Event()
    release_multiview = threading.Event()
    base_attempting = threading.Event()
    base_entered = threading.Event()

    def multiview_worker():
        with _MV_RUN_LOCK:
            multiview_entered.set()
            assert base_attempting.wait(2)
            assert release_multiview.wait(2)
            postprocess.to_glb(texture_size=2048)

    def base_worker():
        base_attempting.set()
        with _scoped_single_view_texture_export(1024, inference):
            base_entered.set()
            postprocess.to_glb(texture_size=4096)

    multiview = threading.Thread(target=multiview_worker, name="multiview")
    base = threading.Thread(target=base_worker, name="base")
    multiview.start()
    assert multiview_entered.wait(2)
    base.start()
    assert base_attempting.wait(2)

    # The base request must not install either process-global override until
    # the multiview request finishes exporting at its own selected size.
    time.sleep(0.05)
    assert not base_entered.is_set()
    assert os.environ[PIXAL3D_TEXTURE_SIZE_ENV] == sentinel
    assert postprocess.to_glb is original_to_glb

    release_multiview.set()
    multiview.join(2)
    base.join(2)
    assert not multiview.is_alive()
    assert not base.is_alive()
    assert calls == [
        ("multiview", sentinel, 2048),
        ("base", "1024", 1024),
    ]
    assert os.environ[PIXAL3D_TEXTURE_SIZE_ENV] == sentinel
    assert postprocess.to_glb is original_to_glb

    # Guard against accidentally leaving the correct scope unused by run_job.
    assert "_scoped_single_view_texture_export(" in inspect.getsource(run_job)
