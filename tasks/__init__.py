"""Tasks package for Healthcare Scheduling graders."""
from tasks.book_appointment import grade as grade_task_1
from tasks.scheduling_conflicts import grade as grade_task_2
from tasks.priority_scheduling import grade as grade_task_3


class TaskGrader:
    """Backward-compatible class wrapper used by inference.py."""
    def __init__(self, env):
        self.env = env

    def grade_task_1(self) -> float:
        return grade_task_1(self.env)

    def grade_task_2(self) -> float:
        return grade_task_2(self.env)

    def grade_task_3(self) -> float:
        return grade_task_3(self.env)
