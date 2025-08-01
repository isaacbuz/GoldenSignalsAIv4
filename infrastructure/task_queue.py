import json
import os
import time
import uuid
from typing import Any, Callable, Dict

import redis
from rq import Connection, Queue, Worker
from rq.job import Job

from src.infrastructure.config_manager import config_manager
from src.infrastructure.monitoring import system_monitoring


class DistributedTaskQueue:
    """
    Advanced Distributed Task Queue for Scalable Processing

    Features:
    - Redis-based distributed task management
    - Asynchronous task execution
    - Retry and error handling
    - Performance monitoring
    """

    def __init__(self):
        # Redis configuration
        redis_host = config_manager.get('task_queue.redis.host', 'localhost')
        redis_port = config_manager.get('task_queue.redis.port', 6379)
        redis_db = config_manager.get('task_queue.redis.db', 0)

        # Create Redis connection
        self.redis_conn = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )

        # Create task queues
        self.high_priority_queue = Queue('high', connection=self.redis_conn)
        self.default_queue = Queue('default', connection=self.redis_conn)
        self.low_priority_queue = Queue('low', connection=self.redis_conn)

    @system_monitoring.trace_function
    def enqueue_task(
        self,
        func: Callable,
        args: tuple = None,
        kwargs: Dict[str, Any] = None,
        priority: str = 'default',
        retry: int = 3
    ) -> Job:
        """
        Enqueue a task for distributed processing

        Args:
            func (Callable): Function to execute
            args (tuple, optional): Positional arguments
            kwargs (Dict[str, Any], optional): Keyword arguments
            priority (str, optional): Task priority
            retry (int, optional): Number of retry attempts

        Returns:
            Job: Enqueued task job
        """
        args = args or ()
        kwargs = kwargs or {}

        # Select queue based on priority
        queue_map = {
            'high': self.high_priority_queue,
            'default': self.default_queue,
            'low': self.low_priority_queue
        }
        selected_queue = queue_map.get(priority, self.default_queue)

        # Enqueue task with retry
        job = selected_queue.enqueue(
            func,
            args=args,
            kwargs=kwargs,
            retry=retry
        )

        return job

    def get_task_status(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieve task status

        Args:
            job_id (str): Unique job identifier

        Returns:
            Dict[str, Any]: Task status information
        """
        job = Job.fetch(job_id, connection=self.redis_conn)

        return {
            'id': job.id,
            'status': job.get_status(),
            'result': job.result,
            'exception': str(job.exc_info) if job.exc_info else None
        }

    def start_worker(self, queues: list = None):
        """
        Start a task queue worker

        Args:
            queues (list, optional): Queues to process
        """
        queues = queues or ['high', 'default', 'low']

        with Connection(self.redis_conn):
            worker = Worker(queues)
            worker.work()

    def clear_queue(self, queue_name: str = 'default'):
        """
        Clear tasks from a specific queue

        Args:
            queue_name (str, optional): Queue to clear
        """
        queue_map = {
            'high': self.high_priority_queue,
            'default': self.default_queue,
            'low': self.low_priority_queue
        }

        selected_queue = queue_map.get(queue_name, self.default_queue)
        selected_queue.empty()

# Singleton instance
task_queue = DistributedTaskQueue()
