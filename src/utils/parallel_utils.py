"""
Parallel Processing Utilities
=============================
Wrappers for joblib parallelization across IC simulations and other tasks.
"""

import numpy as np
from joblib import Parallel, delayed
from typing import Callable, List, Any, Optional
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parallel_map(
    func: Callable,
    items: List[Any],
    n_jobs: int = -1,
    backend: str = 'loky',
    verbose: int = 0,
    desc: str = None,
    show_progress: bool = True
) -> List[Any]:
    """
    Parallel map function with progress bar.

    Parameters
    ----------
    func : Callable
        Function to apply to each item
    items : List[Any]
        Items to process
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    backend : str
        Joblib backend ('loky', 'threading', 'multiprocessing')
    verbose : int
        Verbosity level
    desc : str
        Progress bar description
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    List[Any]
        Results
    """
    if show_progress:
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(func)(item) for item in tqdm(items, desc=desc)
        )
    else:
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(func)(item) for item in items
        )

    return results


def parallel_ic_simulations(
    ic_func: Callable,
    seed_nodes: List[int],
    n_runs_per_seed: int = 50,
    n_jobs: int = -1,
    **ic_kwargs
) -> List[dict]:
    """
    Run IC simulations in parallel.

    Parameters
    ----------
    ic_func : Callable
        IC simulation function (seed_node, **kwargs) -> reach
    seed_nodes : List[int]
        Seed nodes to simulate
    n_runs_per_seed : int
        Number of runs per seed
    n_jobs : int
        Number of parallel jobs
    **ic_kwargs
        Additional arguments for IC function

    Returns
    -------
    List[dict]
        Results for each seed node
    """
    def run_single_seed(seed_node):
        reaches = []
        for run in range(n_runs_per_seed):
            reach = ic_func(seed_node, seed=run, **ic_kwargs)
            reaches.append(reach)

        return {
            "seed_node": seed_node,
            "n_runs": n_runs_per_seed,
            "mean_reach": np.mean(reaches),
            "std_reach": np.std(reaches),
            "median_reach": np.median(reaches),
            "min_reach": np.min(reaches),
            "max_reach": np.max(reaches),
            "reaches": reaches
        }

    logger.info(f"Running IC simulations for {len(seed_nodes)} seeds × {n_runs_per_seed} runs...")
    start_time = time.time()

    results = parallel_map(
        run_single_seed,
        seed_nodes,
        n_jobs=n_jobs,
        desc="IC Simulations"
    )

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.2f}s ({len(seed_nodes) * n_runs_per_seed / elapsed:.1f} runs/sec)")

    return results


def parallel_louvain_runs(
    louvain_func: Callable,
    n_runs: int = 10,
    n_jobs: int = -1,
    **louvain_kwargs
) -> List[dict]:
    """
    Run Louvain community detection in parallel for stability analysis.

    Parameters
    ----------
    louvain_func : Callable
        Louvain function (seed) -> partition dict
    n_runs : int
        Number of runs
    n_jobs : int
        Number of parallel jobs
    **louvain_kwargs
        Additional arguments

    Returns
    -------
    List[dict]
        Partition results for each run
    """
    def run_single(seed):
        partition = louvain_func(seed=seed, **louvain_kwargs)
        return {"seed": seed, "partition": partition}

    results = parallel_map(
        run_single,
        list(range(n_runs)),
        n_jobs=n_jobs,
        desc="Louvain runs"
    )

    return results


class BatchProcessor:
    """
    Process items in batches with progress tracking.
    """

    def __init__(self, batch_size: int = 100, n_jobs: int = -1):
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def process(
        self,
        func: Callable,
        items: List[Any],
        aggregate: bool = True
    ) -> List[Any]:
        """
        Process items in batches.

        Parameters
        ----------
        func : Callable
            Function to apply
        items : List[Any]
            Items to process
        aggregate : bool
            Whether to flatten results

        Returns
        -------
        List[Any]
            Results
        """
        n_batches = (len(items) + self.batch_size - 1) // self.batch_size
        all_results = []

        for i in tqdm(range(n_batches), desc="Batches"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(items))
            batch = items[start_idx:end_idx]

            batch_results = parallel_map(
                func, batch,
                n_jobs=self.n_jobs,
                show_progress=False
            )

            if aggregate:
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)

        return all_results


def get_optimal_n_jobs(task_type: str = 'cpu') -> int:
    """
    Get optimal number of parallel jobs based on task type.

    Parameters
    ----------
    task_type : str
        'cpu' for CPU-bound, 'io' for I/O-bound tasks

    Returns
    -------
    int
        Recommended n_jobs
    """
    import os
    n_cores = os.cpu_count() or 4

    if task_type == 'cpu':
        # Leave one core for system
        return max(1, n_cores - 1)
    elif task_type == 'io':
        # Can use more workers for I/O-bound
        return n_cores * 2
    else:
        return -1  # Use all cores


if __name__ == "__main__":
    # Example usage
    import time

    def example_task(x):
        time.sleep(0.01)  # Simulate work
        return x ** 2

    data = list(range(100))

    # Sequential
    start = time.time()
    seq_results = [example_task(x) for x in data]
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    par_results = parallel_map(example_task, data, n_jobs=-1, desc="Parallel")
    par_time = time.time() - start

    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel: {par_time:.2f}s")
    print(f"Speedup: {seq_time / par_time:.2f}x")
