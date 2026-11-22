"""Prepare approved inference models outside procurement HTTP requests."""

from contextlib import contextmanager
import os
from threading import Lock, Thread

from werkzeug.exceptions import ServiceUnavailable


KINDS = {'semantic', 'rf'}
_state_lock = Lock()
_active = None
_failed = set()


def enabled(config):
    return (os.environ.get('PSMS_RENDER_OCR_BUDGET') == '1'
            and config.get('PSMS_NLP_RUNTIME') == 'retrained_rf'
            and config.get('PSMS_INFERENCE_ENABLED') is True)


def _ready(kind, config):
    release = config.get('_PSMS_INFERENCE_RELEASE_ID')
    if kind == 'semantic':
        from offline_ocr_mapper import semantic_model_ready
        return semantic_model_ready(release)
    from retrained_nlp_runtime import assets_ready
    return assets_ready(config['PSMS_RETRAINED_NLP_DIR'], release)


def _load(kind, config):
    global _active
    from offline_ocr_mapper import load_local_model, model_memory_scope
    from render_resource_logging import measure

    failed = True
    try:
        with measure('model_preparation_' + kind), model_memory_scope():
            if kind == 'semantic':
                load_local_model()
            else:
                from nlp_runtime_selection import runtime
                runtime(config).load_assets(config['PSMS_RETRAINED_NLP_DIR'])
            if not _ready(kind, config):
                raise RuntimeError('MODEL_PREPARATION_UNAVAILABLE')
            failed = False
    except Exception:
        # Keep the safe resource event, without exception or artifact paths.
        pass
    finally:
        with _state_lock:
            if failed:
                _failed.add(kind)
            _active = None


def state(kind, config, *, start=False):
    global _active
    with _state_lock:
        if _active is not None:
            return 'preparing'
        if _ready(kind, config):
            return 'ready'
        if not start:
            return 'failed' if kind in _failed else 'idle'
        _failed.discard(kind)
        _active = kind
        try:
            Thread(target=_load, args=(kind, config), name='psms-model-preparation', daemon=True).start()
        except RuntimeError:
            _active = None
            _failed.add(kind)
            return 'failed'
        return 'preparing'


@contextmanager
def ready_for_processing(kind, config):
    if not enabled(config):
        yield
        return

    from offline_ocr_mapper import model_memory_scope
    lock = model_memory_scope()
    if not lock.acquire(blocking=False):
        raise ServiceUnavailable('Document or model processing is busy. Please try again shortly.')
    try:
        if kind is not None and state(kind, config, start=True) != 'ready':
            raise ServiceUnavailable('The approved model is preparing. Wait for preparation to finish before submitting.')
        yield
    finally:
        lock.release()
