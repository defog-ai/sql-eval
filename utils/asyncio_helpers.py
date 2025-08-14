import asyncio
import threading
import atexit


_background_event_loop = None
_background_loop_thread = None
_background_loop_lock = threading.Lock()


def _start_background_event_loop():
    global _background_event_loop, _background_loop_thread
    _background_event_loop = asyncio.new_event_loop()

    def _run_loop(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _background_loop_thread = threading.Thread(
        target=_run_loop, args=(_background_event_loop,), daemon=True
    )
    _background_loop_thread.start()


def _ensure_background_loop():
    global _background_event_loop
    with _background_loop_lock:
        if _background_event_loop is None or _background_event_loop.is_closed():
            _start_background_event_loop()


def _shutdown_background_event_loop():
    global _background_event_loop, _background_loop_thread
    with _background_loop_lock:
        if _background_event_loop is not None and not _background_event_loop.is_closed():
            _background_event_loop.call_soon_threadsafe(_background_event_loop.stop)
        if _background_loop_thread is not None:
            _background_loop_thread.join(timeout=1.0)
        if _background_event_loop is not None and not _background_event_loop.is_closed():
            _background_event_loop.close()
        _background_event_loop = None
        _background_loop_thread = None


atexit.register(_shutdown_background_event_loop)


def run_coro_sync(coro):
    _ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _background_event_loop)
    return future.result()


def submit_coro(coro):
    _ensure_background_loop()
    return asyncio.run_coroutine_threadsafe(coro, _background_event_loop)


