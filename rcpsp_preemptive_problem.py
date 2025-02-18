import sys, math, random, os
from qubots.base_problem import BaseProblem

class RCPSPPreemptiveProblem(BaseProblem):
    """
    Preemptive Resource-Constrained Project Scheduling Problem (RCPSP) with pseudo-preemption.

    Each task has a fixed duration and requires certain amounts of renewable resources when executed.
    Tasks may be preempted (split into up to a fixed number of subtasks) but their total processing time must equal
    the task’s duration. Precedence constraints enforce that each task must finish before any of its successors begin.
    Additionally, at every time unit, the sum of resource usages for all tasks running at that time must not exceed
    the available resource capacities.
    
    The objective is to minimize the makespan (the maximum finish time among all tasks).
    """
    
    def __init__(self, instance_file: str, **kwargs):
        (self.nb_tasks, self.nb_resources, self.capacity, self.duration, self.weight,
         self.nb_successors, self.successors, self.horizon, self.max_nb_preemptions) = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)


        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        first_line = lines[0].split()
        nb_tasks = int(first_line[0])
        nb_resources = int(first_line[1])
        # Second line: capacities for each resource
        capacity = [int(val) for val in lines[1].split()[:nb_resources]]
        
        duration = [0] * nb_tasks
        weight = [[] for _ in range(nb_tasks)]
        nb_successors = [0] * nb_tasks
        successors = [[] for _ in range(nb_tasks)]
        for i in range(nb_tasks):
            tokens = lines[i+2].split()
            duration[i] = int(tokens[0])
            weight[i] = [int(tokens[r+1]) for r in range(nb_resources)]
            nb_successors[i] = int(tokens[nb_resources+1])
            successors[i] = [int(tokens[nb_resources+2+s]) - 1 for s in range(nb_successors[i])]
        horizon = sum(duration)
        max_nb_preemptions = 4
        return nb_tasks, nb_resources, capacity, duration, weight, nb_successors, successors, horizon, max_nb_preemptions

    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution: a dictionary with key "schedule" mapping to a list of length nb_tasks.
                    Each element is a list of up to max_nb_preemptions intervals (tuples of (start, end)) 
                    representing the execution periods of that task.
                    
        Returns:
          The makespan (maximum end time among all tasks) if all constraints are satisfied; otherwise, a high penalty.
        """
        if not isinstance(solution, dict) or "schedule" not in solution:
            return sys.maxsize
        schedule = solution["schedule"]
        if len(schedule) != self.nb_tasks:
            return sys.maxsize
        
        task_finish = [0] * self.nb_tasks
        # Check each task's intervals.
        for i in range(self.nb_tasks):
            intervals = schedule[i]
            if not intervals or not isinstance(intervals, list):
                return sys.maxsize
            total = 0
            prev_end = None
            for (start, end) in intervals:
                if start > end:
                    return sys.maxsize
                if prev_end is not None and start < prev_end - 1e-6:
                    return sys.maxsize
                total += (end - start)
                prev_end = end
            if abs(total - self.duration[i]) > 1e-6:
                return sys.maxsize
            task_finish[i] = intervals[-1][1]
        
        # Check precedence constraints.
        for i in range(self.nb_tasks):
            for succ in self.successors[i]:
                if task_finish[i] > schedule[succ][0][0] + 1e-6:
                    return sys.maxsize
        
        # Check cumulative resource constraints over integer time steps.
        for t in range(self.horizon):
            for r in range(self.nb_resources):
                usage = 0
                for i in range(self.nb_tasks):
                    for (start, end) in schedule[i]:
                        if t >= start and t < end:
                            usage += self.weight[i][r]
                if usage > self.capacity[r] + 1e-6:
                    return sys.maxsize
        
        makespan = max(task_finish)
        return makespan

    def random_solution(self):
        """
        Generates a simple feasible solution by scheduling each task as a single interval (no preemption).
        Tasks are scheduled in a greedy topological order based on precedence constraints.
        """
        # Compute indegree for topological sort.
        indegree = [0]*self.nb_tasks
        for i in range(self.nb_tasks):
            for succ in self.successors[i]:
                indegree[succ] += 1
        ready = [i for i in range(self.nb_tasks) if indegree[i] == 0]
        order = []
        while ready:
            i = ready.pop(0)
            order.append(i)
            for succ in self.successors[i]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
        # Greedy scheduling: for each task in order, set start time = max(finish time of all predecessors)
        start_time = [0]*self.nb_tasks
        schedule = [None]*self.nb_tasks
        for i in order:
            st = 0
            for j in range(self.nb_tasks):
                if i in self.successors[j]:
                    st = max(st, start_time[j] + self.duration[j])
            schedule[i] = [(st, st + self.duration[i])]
            start_time[i] = st + self.duration[i]
        return {"schedule": schedule}
