# SPDX-License-Identifier: Apache-2.0
"""``_poll_transfer`` must read a completed future's result exactly once.

The admission gate re-polls held retrieves on every engine step; each
``result()`` call re-synchronizes the device event, and on NPU that
synchronize blocks on the stream. Memoizing the result on the future keeps
the re-poll cheap and the engine step short.
"""

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import _poll_transfer


class _FakeFuture:
    def __init__(self):
        self.result_calls = 0
        self.result = self._result_impl
        self._value = {"tokens": 42}

    def _result_impl(self, timeout=None):
        self.result_calls += 1
        return self._value

    def query(self):
        return True


def test_completed_future_result_read_once():
    fut = _FakeFuture()
    ok1, r1 = _poll_transfer("rid", fut)
    ok2, r2 = _poll_transfer("rid", fut)
    ok3, r3 = _poll_transfer("rid", fut)
    assert ok1 and ok2 and ok3
    assert r1 is r2 is r3
    assert fut.result_calls == 1


def test_not_done_future_polls_again_without_memo():
    fut = _FakeFuture()
    fut.query = lambda: False
    ok, r = _poll_transfer("rid", fut)
    assert ok is False and r is None
    assert fut.result_calls == 0
    # Once it completes, the normal path runs and memoizes.
    ok2, r2 = _poll_transfer("rid", fut)
    ok3, r3 = _poll_transfer("rid", fut)
    assert ok2 and ok3 and r2 is r3
    assert fut.result_calls == 1


def test_falsy_result_is_still_memoized():
    """A falsy result (None payload) must not look like 'never read'."""
    fut = _FakeFuture()
    fut._value = None
    ok1, r1 = _poll_transfer("rid", fut)
    ok2, r2 = _poll_transfer("rid", fut)
    assert ok1 and ok2 and r1 is None and r2 is None
    assert fut.result_calls == 1
