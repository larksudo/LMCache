# SPDX-License-Identifier: Apache-2.0
"""
Thread pool with affinity routing.

Tasks submitted with the same ``affinity_key`` always execute on the same
worker thread.  Within each worker, tasks execute sequentially in FIFO order.

This is used for GPU-bound request handlers (STORE / RETRIEVE) so that all
operations for a given vLLM instance land on one thread, eliminating the need
for per-instance locks on the shared temporary GPU buffer.

Queue-wait timing
-----------------
Every task is stamped with its enqueue time and the queue depth at that moment;
the worker computes the wait on dequeue and publishes it three ways:

* on the returned :class:`~concurrent.futures.Future` as ``queue_wait_ms`` /
  ``queue_depth`` / ``slot``, for the transport layer that holds the future;
* in thread-locals readable through :func:`current_queue_wait_ms` and friends,
  so a handler can attribute the wait to its own ``request_id`` without any
  change to its signature;
* in a periodic aggregate INFO line.

This exists because with ``--max-gpu-workers 1`` every client key maps to slot
``next_slot % 1 == 0``, so all ranks share one thread and their handlers are
serialized. A retrieve duration measured by the engine then contains an
unmeasured queueing component that depends on *other* requests' work, not on
its own size -- which would make any duration model built from per-request
attributes systematically wrong. Separating the two is what makes that
distinguishable from a transfer-bandwidth limit.
"""

# Standard
from concurrent.futures import Future
import queue
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Sentinel object to signal worker shutdown
_SHUTDOWN = object()

# Number of completed tasks between aggregate timing log lines (0 = never).
_TIMING_LOG_EVERY = 2000

# Published to the handler running on this thread. Handlers read it through the
# accessors below rather than touching the local directly.
_task_timing = threading.local()


def current_queue_wait_ms() -> float:
    """Queue wait of the task running on this thread, in milliseconds.

    Returns:
        The enqueue-to-dequeue wait, or ``-1.0`` when the caller is not inside
        an :class:`AffinityThreadPool` worker. The sentinel lets handlers tell
        "not measured" apart from "measured as zero", since handlers also run
        on the normal and serialized executors.
    """
    return getattr(_task_timing, "queue_wait_ms", -1.0)


def current_queue_depth() -> int:
    """Tasks still queued when the current one was dequeued, or ``-1``."""
    return getattr(_task_timing, "queue_depth", -1)


def current_slot() -> int:
    """Worker slot running the current task, or ``-1``."""
    return getattr(_task_timing, "slot", -1)


class AffinityThreadPool:
    """Thread pool that routes tasks to workers by affinity key.

    Not thread-safe: the request transport must serialize calls to ``submit()``.
    ZMQ submits from its main loop, while gRPC protects submission with a lock.

    Args:
        max_workers: Number of worker threads.
        thread_name_prefix: Prefix for worker thread names.
    """

    def __init__(
        self,
        max_workers: int,
        thread_name_prefix: str = "affinity",
    ) -> None:
        self._num_workers = max_workers
        self._queues: list[queue.Queue] = [queue.Queue() for _ in range(max_workers)]
        self._threads: list[threading.Thread] = []
        # Maps an affinity_key -> the worker slot (thread index) bound to it.
        self._key_to_slot: dict[int, int] = {}
        self._next_slot = 0
        self._overflow_warned = False
        # Aggregate queue-wait stats for the periodic log line. Guarded by the
        # GIL for these simple accumulations; they are diagnostics, so a torn
        # read would only perturb a reported average, never a task.
        self._timed_tasks = 0
        self._timed_wait_total = 0.0
        self._timed_wait_max = 0.0
        self._timed_depth_max = 0
        self._thread_name_prefix = thread_name_prefix
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self._queues[i], i),
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()
            self._threads.append(t)

        logger.info(
            "Created AffinityThreadPool '%s' with %d worker slots: up to %d "
            "distinct affinity keys each bind to their own thread before slots "
            "are shared. Compare this against the number of clients expected to "
            "connect to confirm routing.",
            thread_name_prefix,
            max_workers,
            max_workers,
        )

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker(self, q: queue.Queue, slot: int) -> None:
        while True:
            item = q.get()
            # The shutdown sentinel is not a task tuple; check it before
            # unpacking or the worker dies on a TypeError at shutdown.
            if item is _SHUTDOWN:
                break
            future, fn, args, kwargs, t_enq, depth = item
            t_deq = time.perf_counter()
            wait_ms = (t_deq - t_enq) * 1000.0

            # Publish for the handler (see the accessors at module scope)
            # and for the transport layer that owns the future. Both are
            # cleared in the finally block so a reused thread never leaks a
            # stale value into an unrelated handler.
            _task_timing.queue_wait_ms = wait_ms
            _task_timing.queue_depth = depth
            _task_timing.slot = slot
            future.queue_wait_ms = wait_ms  # type: ignore[attr-defined]
            future.queue_depth = depth  # type: ignore[attr-defined]
            future.slot = slot  # type: ignore[attr-defined]
            self._note_timing(wait_ms, depth)

            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except BaseException as exc:
                    future.set_exception(exc)
                finally:
                    # Clear the thread-locals so a reused thread never leaks a
                    # stale timing into an unrelated handler.
                    _task_timing.queue_wait_ms = -1.0
                    _task_timing.queue_depth = -1
                    _task_timing.slot = -1

    def _note_timing(self, wait_ms: float, depth: int) -> None:
        """Accumulate one task's queue wait; log an aggregate periodically."""
        self._timed_tasks += 1
        self._timed_wait_total += wait_ms
        if wait_ms > self._timed_wait_max:
            self._timed_wait_max = wait_ms
        if depth > self._timed_depth_max:
            self._timed_depth_max = depth
        if _TIMING_LOG_EVERY and self._timed_tasks % _TIMING_LOG_EVERY == 0:
            logger.info(
                "affinity_pool '%s': tasks=%d slots=%d queue_wait_ms("
                "mean=%.2f max=%.2f) queue_depth_max=%d",
                self._thread_name_prefix,
                self._timed_tasks,
                self._num_workers,
                self._timed_wait_total / self._timed_tasks,
                self._timed_wait_max,
                self._timed_depth_max,
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _slot_for_key(self, affinity_key: int) -> int:
        """Return the worker slot bound to ``affinity_key``, assigning on first use.

        Returns:
            The worker slot (an index in ``[0, _num_workers)``) for the key.
        """
        slot = self._key_to_slot.get(affinity_key)
        if slot is not None:
            return slot

        # First time we see this key -- bind it to the next free slot.
        slot = self._next_slot % self._num_workers
        is_overflow = self._next_slot >= self._num_workers
        self._key_to_slot[affinity_key] = slot
        self._next_slot += 1

        logger.info(
            "AffinityThreadPool: affinity_key=%d assigned to worker "
            "slot %d of %d (thread %s); %d distinct key(s) now bound",
            affinity_key,
            slot,
            self._num_workers,
            self._threads[slot].name,
            len(self._key_to_slot),
        )
        if is_overflow and not self._overflow_warned:
            self._overflow_warned = True
            logger.warning(
                "AffinityThreadPool: affinity_key=%d wrapped onto worker slot "
                "%d (only %d workers), so it shares a thread with an earlier "
                "key and their tasks are serialized. Increase the worker count "
                "to give each client its own thread.",
                affinity_key,
                slot,
                self._num_workers,
            )
        return slot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn, *args, affinity_key: int = 0, **kwargs) -> Future:
        """Submit *fn* for execution on the worker bound to *affinity_key*.

        Returns:
            A :class:`concurrent.futures.Future` that carries three extra
            attributes once its task starts: ``queue_wait_ms``,
            ``queue_depth`` and ``slot``. They are absent before then, so
            callers should read them with
            ``getattr(future, "queue_wait_ms", None)``.
        """
        future: Future = Future()
        slot = self._slot_for_key(affinity_key)
        q = self._queues[slot]
        # qsize() is approximate under concurrency but adequate here: it
        # reports how much work was already waiting, which is what makes the
        # queueing visible at all.
        q.put((future, fn, args, kwargs, time.perf_counter(), q.qsize()))
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool.

        Sends a shutdown sentinel to every worker.  If *wait* is true, blocks
        until all workers have exited.
        """
        for q in self._queues:
            q.put(_SHUTDOWN)
        if wait:
            for t in self._threads:
                t.join()
