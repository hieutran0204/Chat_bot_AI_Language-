# name: settings.py
# description: arq WorkerSettings configuring task registry, cron schedule, and concurrency.

from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.worker.tasks import process_conversation_job, scan_idle_conversations


class WorkerSettings:
    """arq worker configuration settings."""

    functions = [process_conversation_job]
    cron_jobs = [
        cron(
            scan_idle_conversations,
            minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55},
        )
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.effective_arq_max_jobs
    job_timeout = settings.arq_job_timeout
    keep_result = 0  # Do not retain job results; allow re-enqueueing with the same job_id
    max_tries = settings.arq_max_tries
