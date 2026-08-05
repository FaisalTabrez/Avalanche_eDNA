"""
Task Manager for Background Process Orchestration
Handles task lifecycle, progress tracking, and persistence across navigation
"""

import json
import os
import queue
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class TaskStatus(Enum):
    """Task execution status"""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskType(Enum):
    """Types of tasks that can be executed"""

    ANALYSIS = "analysis"
    TRAINING = "training"
    EMBEDDING = "embedding"
    CLUSTERING = "clustering"
    TAXONOMY = "taxonomy"
    DOWNLOAD = "download"


@dataclass
class TaskInfo:
    """Information about a background task"""

    task_id: str
    task_type: TaskType
    name: str
    status: TaskStatus
    progress: float  # 0-100
    stage: str
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    dataset_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None

    # Runtime metrics
    elapsed_time: float = 0.0  # seconds
    estimated_time_remaining: Optional[float] = None

    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0

    def to_dict(self):
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        """Create TaskInfo from dictionary"""
        data["task_type"] = TaskType(data["task_type"])
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


class TaskManager:
    """
    Manages background tasks with progress tracking and persistence
    """

    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize task manager

        Args:
            state_dir: Directory to store task state (defaults to data/task_state)
        """
        self.state_dir = Path(state_dir or "data/task_state")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.tasks: Dict[str, TaskInfo] = {}
        self.progress_queues: Dict[str, Queue] = {}
        self.stop_events: Dict[str, Event] = {}
        self.processes: Dict[str, Process] = {}

        # Load persisted tasks
        self._load_tasks()

        # Clean up stale processes
        self._cleanup_stale_tasks()

    def create_task(
        self,
        task_type: TaskType,
        name: str,
        dataset_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new task

        Args:
            task_type: Type of task
            name: Display name for task
            dataset_name: Optional dataset name
            config: Task configuration

        Returns:
            task_id: Unique task identifier
        """
        task_id = f"{task_type.value}_{int(time.time() * 1000)}"

        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            name=name,
            status=TaskStatus.QUEUED,
            progress=0.0,
            stage="Initializing",
            message="Task created",
            created_at=datetime.now().isoformat(),
            dataset_name=dataset_name,
            config=config or {},
        )

        self.tasks[task_id] = task
        self._save_task(task_id)

        return task_id

    def start_task(
        self,
        task_id: str,
        target_func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Start executing a task in background process

        Args:
            task_id: Task to start
            target_func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        if task.status == TaskStatus.RUNNING:
            raise ValueError(f"Task {task_id} is already running")

        # Create communication channels
        progress_queue = Queue()
        stop_event = Event()

        self.progress_queues[task_id] = progress_queue
        self.stop_events[task_id] = stop_event

        # Prepare kwargs with progress callback
        kwargs = kwargs or {}
        kwargs["progress_queue"] = progress_queue
        kwargs["stop_event"] = stop_event
        kwargs["task_id"] = task_id

        # Start background process
        process = Process(
            target=self._task_wrapper,
            args=(target_func, progress_queue, stop_event, task_id, args, kwargs),
        )
        process.start()

        self.processes[task_id] = process

        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        self._save_task(task_id)

        # Start progress monitor thread
        monitor_thread = threading.Thread(
            target=self._monitor_progress, args=(task_id,), daemon=True
        )
        monitor_thread.start()

    def _task_wrapper(
        self,
        target_func: Callable,
        progress_queue: Queue,
        stop_event: Event,
        task_id: str,
        args: tuple,
        kwargs: Dict[str, Any],
    ):
        """Wrapper to execute task and handle errors"""
        try:
            result = target_func(*args, **kwargs)
            progress_queue.put({"type": "complete", "result": result})
        except Exception as e:
            progress_queue.put(
                {"type": "error", "error": str(e), "traceback": traceback.format_exc()}
            )

    def _monitor_progress(self, task_id: str):
        """Monitor task progress from queue"""
        progress_queue = self.progress_queues.get(task_id)
        if not progress_queue:
            return

        task = self.tasks.get(task_id)
        if not task:
            return

        start_time = time.time()

        while True:
            try:
                # Check for updates (timeout to allow periodic checks)
                try:
                    update = progress_queue.get(timeout=1.0)
                except queue.Empty:
                    # Update elapsed time
                    task.elapsed_time = time.time() - start_time
                    self._save_task(task_id)
                    continue

                if update["type"] == "progress":
                    task.progress = update.get("progress", task.progress)
                    task.stage = update.get("stage", task.stage)
                    task.message = update.get("message", task.message)

                    # Update metrics
                    task.cpu_percent = update.get("cpu_percent", 0.0)
                    task.memory_mb = update.get("memory_mb", 0.0)
                    task.gpu_memory_mb = update.get("gpu_memory_mb", 0.0)
                    task.estimated_time_remaining = update.get("eta", None)

                elif update["type"] == "complete":
                    task.status = TaskStatus.COMPLETED
                    task.progress = 100.0
                    task.stage = "Completed"
                    task.message = "Task completed successfully"
                    task.completed_at = datetime.now().isoformat()
                    task.results = update.get("result")
                    self._save_task(task_id)
                    break

                elif update["type"] == "error":
                    task.status = TaskStatus.FAILED
                    task.stage = "Failed"
                    task.error = update.get("error")
                    task.message = f"Error: {update.get('error', 'Unknown error')}"
                    task.completed_at = datetime.now().isoformat()
                    self._save_task(task_id)
                    break

                # Update elapsed time
                task.elapsed_time = time.time() - start_time
                self._save_task(task_id)

            except Exception as e:
                print(f"Error monitoring task {task_id}: {e}")
                break

    def pause_task(self, task_id: str):
        """Pause a running task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        if task.status != TaskStatus.RUNNING:
            raise ValueError(f"Task {task_id} is not running")

        # Signal stop (resume requires restart)
        if task_id in self.stop_events:
            self.stop_events[task_id].set()

        task.status = TaskStatus.PAUSED
        task.message = "Task paused by user"
        self._save_task(task_id)

    def stop_task(self, task_id: str):
        """Stop a running task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        # Signal stop
        if task_id in self.stop_events:
            self.stop_events[task_id].set()

        # Terminate process if still running
        if task_id in self.processes:
            process = self.processes[task_id]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
            del self.processes[task_id]

        task.status = TaskStatus.STOPPED
        task.completed_at = datetime.now().isoformat()
        task.message = "Task stopped by user"
        self._save_task(task_id)

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[TaskInfo]:
        """Get all tasks"""
        return list(self.tasks.values())

    def get_active_tasks(self) -> List[TaskInfo]:
        """Get all active (running/paused/queued) tasks"""
        return [
            task
            for task in self.tasks.values()
            if task.status in [TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.QUEUED]
        ]

    def remove_task(self, task_id: str):
        """Remove a task (only if not running)"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        if task.status == TaskStatus.RUNNING:
            raise ValueError(f"Cannot remove running task {task_id}. Stop it first.")

        # Clean up
        del self.tasks[task_id]
        task_file = self.state_dir / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink()

    def clear_completed_tasks(self):
        """Remove all completed/failed/stopped tasks"""
        to_remove = [
            task_id
            for task_id, task in self.tasks.items()
            if task.status
            in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED]
        ]
        for task_id in to_remove:
            self.remove_task(task_id)

    def _save_task(self, task_id: str):
        """Persist task state to disk"""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task_file = self.state_dir / f"{task_id}.json"

        with open(task_file, "w") as f:
            json.dump(task.to_dict(), f, indent=2)

    def _load_tasks(self):
        """Load persisted tasks from disk"""
        if not self.state_dir.exists():
            return

        for task_file in self.state_dir.glob("*.json"):
            try:
                with open(task_file, "r") as f:
                    data = json.load(f)
                    task = TaskInfo.from_dict(data)
                    self.tasks[task.task_id] = task
            except Exception as e:
                print(f"Error loading task from {task_file}: {e}")

    def _cleanup_stale_tasks(self):
        """Clean up tasks that were running but process died"""
        for task_id, task in list(self.tasks.items()):
            if task.status == TaskStatus.RUNNING:
                # Process not in memory, mark as failed
                task.status = TaskStatus.FAILED
                task.error = "Process terminated unexpectedly (app restart or crash)"
                task.message = "Task failed due to process termination"
                task.completed_at = datetime.now().isoformat()
                self._save_task(task_id)


# Global task manager instance
_task_manager = None


def get_task_manager() -> TaskManager:
    """Get global task manager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
