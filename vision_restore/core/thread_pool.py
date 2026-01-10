"""
Vision-Restore AI - Thread Pool Manager

Manages background processing threads for batch operations and non-blocking UI.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import time

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of a processing task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    task_id: str
    input_path: Path
    output_path: Path
    settings: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class ThreadPoolManager:
    """Manages a pool of worker threads for batch processing."""

    def __init__(self, max_workers: int = 2):
        """Initialize the thread pool manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.tasks: Dict[str, ProcessingTask] = {}
        self.task_queue: queue.Queue = queue.Queue()
        self.futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._paused = False
        
        # Callbacks
        self.on_task_started: Optional[Callable[[ProcessingTask], None]] = None
        self.on_task_progress: Optional[Callable[[ProcessingTask], None]] = None
        self.on_task_completed: Optional[Callable[[ProcessingTask], None]] = None
        self.on_task_failed: Optional[Callable[[ProcessingTask], None]] = None
        self.on_all_completed: Optional[Callable[[], None]] = None

    def start(self) -> None:
        """Start the thread pool."""
        if self.executor is not None:
            return
        
        self._stop_event.clear()
        self._paused = False
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"Thread pool started with {self.max_workers} workers")

    def stop(self) -> None:
        """Stop the thread pool and cancel pending tasks."""
        self._stop_event.set()
        
        if self.executor:
            # Cancel pending futures
            for task_id, future in self.futures.items():
                if not future.done():
                    future.cancel()
                    if task_id in self.tasks:
                        self.tasks[task_id].status = TaskStatus.CANCELLED
            
            self.executor.shutdown(wait=False)
            self.executor = None
        
        logger.info("Thread pool stopped")

    def pause(self) -> None:
        """Pause processing of new tasks."""
        self._paused = True
        logger.info("Processing paused")

    def resume(self) -> None:
        """Resume processing of tasks."""
        self._paused = False
        logger.info("Processing resumed")

    @property
    def is_paused(self) -> bool:
        """Check if processing is paused."""
        return self._paused

    def add_task(
        self,
        task_id: str,
        input_path: Path,
        output_path: Path,
        processor: Callable[[Path, Path, Callable[[float, str], None]], Any],
        settings: Optional[Dict[str, Any]] = None,
    ) -> ProcessingTask:
        """Add a new task to the queue.
        
        Args:
            task_id: Unique identifier for the task
            input_path: Path to input image
            output_path: Path for output image
            processor: Processing function to execute
            settings: Optional processing settings
            
        Returns:
            The created task object
        """
        task = ProcessingTask(
            task_id=task_id,
            input_path=input_path,
            output_path=output_path,
            settings=settings or {},
        )
        
        with self._lock:
            self.tasks[task_id] = task
        
        # Submit to executor
        if self.executor:
            future = self.executor.submit(
                self._process_task,
                task,
                processor,
            )
            self.futures[task_id] = future
        
        logger.debug(f"Task added: {task_id}")
        return task

    def _process_task(
        self,
        task: ProcessingTask,
        processor: Callable[[Path, Path, Callable[[float, str], None]], Any],
    ) -> Any:
        """Execute a processing task.
        
        Args:
            task: The task to process
            processor: Processing function
        """
        # Wait if paused
        while self._paused and not self._stop_event.is_set():
            time.sleep(0.1)
        
        if self._stop_event.is_set():
            task.status = TaskStatus.CANCELLED
            return None
        
        try:
            # Update status
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            if self.on_task_started:
                self.on_task_started(task)
            
            # Progress callback
            def progress_callback(progress: float, message: str):
                task.progress = progress
                task.message = message
                if self.on_task_progress:
                    self.on_task_progress(task)
            
            # Execute processor
            result = processor(task.input_path, task.output_path, progress_callback)
            
            # Mark completed
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.result = result
            task.completed_at = time.time()
            
            if self.on_task_completed:
                self.on_task_completed(task)
            
            logger.info(f"Task completed: {task.task_id}")
            
            # Check if all tasks are done
            self._check_all_completed()
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            logger.error(f"Task failed: {task.task_id} - {e}")
            
            if self.on_task_failed:
                self.on_task_failed(task)
            
            self._check_all_completed()
            
            return None

    def _check_all_completed(self) -> None:
        """Check if all tasks are completed and trigger callback."""
        with self._lock:
            pending = [
                t for t in self.tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]
        
        if not pending and self.on_all_completed:
            self.on_all_completed()

    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get a task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task object or None if not found
        """
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[ProcessingTask]:
        """Get all tasks.
        
        Returns:
            List of all task objects
        """
        with self._lock:
            return list(self.tasks.values())

    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        with self._lock:
            return len([
                t for t in self.tasks.values()
                if t.status == TaskStatus.PENDING
            ])

    def get_completed_count(self) -> int:
        """Get number of completed tasks."""
        with self._lock:
            return len([
                t for t in self.tasks.values()
                if t.status == TaskStatus.COMPLETED
            ])

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            
            # Cancel future if it exists
            future = self.futures.get(task_id)
            if future and not future.done():
                future.cancel()
            
            logger.info(f"Task cancelled: {task_id}")
            return True
        
        return False

    def clear_completed(self) -> None:
        """Remove completed and failed tasks from the list."""
        with self._lock:
            to_remove = [
                task_id for task_id, task in self.tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            
            for task_id in to_remove:
                del self.tasks[task_id]
                if task_id in self.futures:
                    del self.futures[task_id]
        
        logger.debug(f"Cleared {len(to_remove)} completed tasks")

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            tasks = list(self.tasks.values())
        
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        
        total_time = 0
        for task in completed:
            if task.started_at and task.completed_at:
                total_time += task.completed_at - task.started_at
        
        return {
            "total_tasks": len(tasks),
            "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
            "running": len([t for t in tasks if t.status == TaskStatus.RUNNING]),
            "completed": len(completed),
            "failed": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "cancelled": len([t for t in tasks if t.status == TaskStatus.CANCELLED]),
            "average_time": total_time / len(completed) if completed else 0,
            "total_time": total_time,
        }
