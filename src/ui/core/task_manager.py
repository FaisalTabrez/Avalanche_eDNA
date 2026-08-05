import json
import multiprocessing
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


class TaskManager:
    """
    Manages background tasks for the Avalanche system.
    Uses multiprocessing to run tasks without blocking the UI.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance.tasks = {}
            cls._instance.processes = {}
            # Ensure storage directory exists
            cls._instance.storage_path = Path("data/tasks")
            cls._instance.storage_path.mkdir(parents=True, exist_ok=True)
            cls._instance.load_tasks()
        return cls._instance

    def load_tasks(self):
        """Load task history from disk."""
        try:
            history_file = self.storage_path / "history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    self.tasks = json.load(f)
        except Exception as e:
            print(f"Error loading task history: {e}")
            self.tasks = {}

    def save_tasks(self):
        """Save task history to disk."""
        try:
            history_file = self.storage_path / "history.json"
            with open(history_file, "w") as f:
                json.dump(self.tasks, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving task history: {e}")

    def submit_task(
        self, name: str, target_func, args: tuple = (), kwargs: dict = {}
    ) -> str:
        """
        Submit a new task to run in the background.
        Returns the task ID.
        """
        task_id = str(uuid.uuid4())

        # Create a wrapper to handle status updates and logging
        process = multiprocessing.Process(
            target=self._task_wrapper, args=(task_id, name, target_func, args, kwargs)
        )

        self.tasks[task_id] = {
            "id": task_id,
            "name": name,
            "status": "queued",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "pid": None,
            "error": None,
        }

        process.start()
        self.processes[task_id] = process

        # Update with PID
        self.tasks[task_id]["pid"] = process.pid
        self.tasks[task_id]["status"] = "running"
        self.save_tasks()

        return task_id

    def _task_wrapper(self, task_id, name, func, args, kwargs):
        """Internal wrapper to run the function and capture output/errors."""
        # In a real implementation, we would redirect stdout/stderr to a file
        try:
            # Update status file (since we can't share memory easily across processes without a Manager)
            # For simplicity in this UI demo, we'll assume the main process checks process liveness
            # But to persist 'completed' state, we should write to a specific task file

            result = func(*args, **kwargs)
            self._update_task_file(task_id, "completed", result=str(result))

        except Exception as e:
            self._update_task_file(task_id, "failed", error=str(e))

    def _update_task_file(self, task_id, status, result=None, error=None):
        """Worker process updates its own status file."""
        task_file = self.storage_path / f"{task_id}.json"
        data = {
            "status": status,
            "end_time": datetime.now().isoformat(),
            "result": result,
            "error": error,
        }
        with open(task_file, "w") as f:
            json.dump(data, f)

    def get_task_status(self, task_id: str) -> Dict:
        """Get the current status of a task."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        # Check if there's an update file from the worker
        task_file = self.storage_path / f"{task_id}.json"
        if task_file.exists():
            try:
                with open(task_file, "r") as f:
                    update = json.load(f)
                    task.update(update)
                    # Clean up process reference if done
                    if (
                        task["status"] in ["completed", "failed"]
                        and task_id in self.processes
                    ):
                        del self.processes[task_id]
                    self.save_tasks()
            except:
                pass

        # Check if process is still alive if marked running
        if task["status"] == "running" and task_id in self.processes:
            if not self.processes[task_id].is_alive():
                # It died without writing to file?
                if not task_file.exists():
                    task["status"] = "failed"
                    task["error"] = "Process terminated unexpectedly"
                    self.save_tasks()

        return task

    def stop_task(self, task_id: str):
        """Stop a running task."""
        if task_id in self.processes:
            process = self.processes[task_id]
            process.terminate()
            process.join()
            del self.processes[task_id]

            self.tasks[task_id]["status"] = "stopped"
            self.tasks[task_id]["end_time"] = datetime.now().isoformat()
            self.save_tasks()
            return True
        return False

    def get_all_tasks(self):
        """Get all tasks, updating their status first."""
        for task_id in list(self.tasks.keys()):
            self.get_task_status(task_id)
        return list(self.tasks.values())

    def get_system_metrics(self):
        """Get current system resource usage."""
        return {
            "cpu": psutil.cpu_percent(interval=None),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
        }
