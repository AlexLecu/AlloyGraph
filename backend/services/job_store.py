"""Background-job store for long-running agent runs.

Why this exists
---------------
A full Analyst -> Reviewer evaluation takes 100-200 s (two 70B round-trips plus
KG search). Holding an HTTP request open that long worked only as long as every
hop allowed it, and the public deployment sits behind the CloudUT reverse proxy,
which gives up at roughly 60 s and returns 504. Raising our own timeouts fixed
the layers we control but cannot fix theirs, so no request may stay open that
long any more: the client submits a job, gets an id in well under a second, and
polls for the outcome.

Why SQLite and not a dict
-------------------------
gunicorn runs 4 *sync* workers -- separate processes, no shared memory. A plain
module-level dict would put job 7 in worker 2's memory, and the status poll that
landed on worker 3 would return 404 about three times out of four. The
alternatives were --preload with a single worker (serialises every request in
the app, including /health) or a shared store. SQLite is in the stdlib, needs no
new service, and handles multi-process access correctly in WAL mode.

Durability is deliberately not a goal. The database lives at
$ALLOYGRAPH_JOB_DB (default /tmp/alloygraph_jobs.db), inside the container's
filesystem, so a restart drops every job. That is the right trade: an in-flight
job dies with the worker that was running it anyway, and a client polling a job
lost to a restart gets a clean 404 rather than a row that will never progress.
"""

import json
import logging
import os
import sqlite3
import threading
import time
import uuid

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "/tmp/alloygraph_jobs.db"

#: Finished jobs (done/failed) are deleted this long after they finished. The
#: frontend polls every 2.5 s and stops on the first terminal status, so an hour
#: is far longer than any client needs and still lets a person re-read a result
#: after stepping away.
FINISHED_TTL_SECONDS = 3600

#: A job still marked "running" this long after it started is presumed dead:
#: its worker was killed, recycled, or the container restarted mid-run. Without
#: this, such a row stays "running" forever and the UI spins indefinitely. The
#: ceiling is well above the ~200 s a real evaluation takes, so a slow-but-alive
#: run is never reaped.
STALE_RUNNING_SECONDS = 1800

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = (STATUS_DONE, STATUS_FAILED)

_init_lock = threading.Lock()
_initialised = False


def _db_path():
    return os.environ.get("ALLOYGRAPH_JOB_DB", DEFAULT_DB_PATH)


def _connect():
    """A short-lived connection.

    One connection per operation rather than a shared handle: sqlite3 objects
    are not safe to pass between threads, and every background job runs in its
    own thread. ``timeout`` is the busy-wait for a write lock -- with WAL and
    sub-millisecond writes, contention between 4 workers is brief, but a hard
    "database is locked" failure would surface as a lost job.
    """
    conn = sqlite3.connect(_db_path(), timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the schema and switch on WAL. Safe to call repeatedly."""
    global _initialised
    with _init_lock:
        if _initialised:
            return
        conn = _connect()
        try:
            # WAL is what makes concurrent access from 4 worker processes work:
            # readers do not block the writer and vice versa. It is a property
            # of the database file, so setting it once is enough, but it is
            # cheap and idempotent to assert on every start.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id          TEXT PRIMARY KEY,
                    kind        TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    created_at  REAL NOT NULL,
                    started_at  REAL,
                    finished_at REAL,
                    request     TEXT,
                    result      TEXT,
                    error       TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_finished "
                "ON jobs (status, finished_at)"
            )
            conn.commit()
            _initialised = True
            logger.info("Job store ready at %s", _db_path())
        finally:
            conn.close()


def create_job(kind, request_payload=None):
    """Insert a pending job and return its id."""
    init_db()
    job_id = uuid.uuid4().hex
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO jobs (id, kind, status, created_at, request) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, kind, STATUS_PENDING, time.time(),
             json.dumps(request_payload) if request_payload is not None else None),
        )
        conn.commit()
    finally:
        conn.close()
    return job_id


def mark_running(job_id):
    conn = _connect()
    try:
        conn.execute(
            "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
            (STATUS_RUNNING, time.time(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_done(job_id, result):
    conn = _connect()
    try:
        conn.execute(
            "UPDATE jobs SET status = ?, finished_at = ?, result = ? WHERE id = ?",
            (STATUS_DONE, time.time(), json.dumps(result), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_failed(job_id, error_message):
    conn = _connect()
    try:
        conn.execute(
            "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
            (STATUS_FAILED, time.time(), str(error_message), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_job(job_id):
    """The job as a dict, or None if there is no such id.

    ``result`` is decoded back into an object; a row whose JSON fails to decode
    is reported as failed rather than raising, so one corrupt row cannot take
    down the status endpoint.
    """
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, kind, status, created_at, started_at, finished_at, "
            "result, error FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    job = dict(row)
    if job.get("result"):
        try:
            job["result"] = json.loads(job["result"])
        except (TypeError, ValueError):
            logger.error("Job %s has undecodable result JSON", job_id)
            job["status"] = STATUS_FAILED
            job["result"] = None
            job["error"] = "Stored result could not be decoded."
    else:
        job["result"] = None
    return job


def cleanup(now=None):
    """Delete expired finished jobs and fail jobs whose worker died.

    Returns ``(deleted, reaped)``. Called opportunistically on job submission
    rather than from a timer: there is no scheduler in a sync gunicorn worker,
    and tying it to submission means the table is tidied exactly when it grows.
    """
    init_db()
    now = now if now is not None else time.time()
    conn = _connect()
    try:
        deleted = conn.execute(
            "DELETE FROM jobs WHERE status IN (?, ?) AND "
            "COALESCE(finished_at, created_at) < ?",
            (STATUS_DONE, STATUS_FAILED, now - FINISHED_TTL_SECONDS),
        ).rowcount

        # A pending job is also eligible: if the thread never started (worker
        # died between INSERT and spawn) it would otherwise sit pending forever.
        reaped = conn.execute(
            "UPDATE jobs SET status = ?, finished_at = ?, error = ? "
            "WHERE status IN (?, ?) AND COALESCE(started_at, created_at) < ?",
            (STATUS_FAILED, now,
             "The worker running this job stopped before it finished. "
             "Please submit it again.",
             STATUS_RUNNING, STATUS_PENDING, now - STALE_RUNNING_SECONDS),
        ).rowcount
        conn.commit()
    finally:
        conn.close()

    if deleted or reaped:
        logger.info("Job store cleanup: %d deleted, %d reaped as stale",
                    deleted, reaped)
    return deleted, reaped


def run_in_background(job_id, fn, *args, **kwargs):
    """Run ``fn`` in a daemon thread, recording the outcome against ``job_id``.

    Any exception is captured as the job's error rather than escaping into a
    thread that nobody is watching -- an unhandled exception here would leave
    the job "running" until the stale reaper caught it half an hour later.

    daemon=True so a shutting-down worker is not held open by an in-flight
    evaluation; the job is then reaped as stale, which is the honest outcome.
    """
    def _target():
        mark_running(job_id)
        try:
            result = fn(*args, **kwargs)
            mark_done(job_id, result)
            logger.info("Job %s finished", job_id)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            logger.exception("Job %s failed", job_id)
            mark_failed(job_id, exc)

    thread = threading.Thread(target=_target, name=f"job-{job_id[:8]}", daemon=True)
    thread.start()
    return thread
