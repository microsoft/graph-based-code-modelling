import multiprocessing
from typing import List, Iterable, Callable, TypeVar

from dpu_utils.utils import RichPath

JobType = TypeVar("JobType")
ResultType = TypeVar("ResultType")


def predictable_shuffle(seq: List)-> List:
    """
    Returns a shuffled version of "seq" that looks random but is in fact
    deterministic. Useful for e.g. selecting an arbitrary subset of files in a
    directory that will be the same between runs.
    :param seq: a sequence
    """
    def _predictable_shuffle(lst, swap):
        sz = len(lst)
        if sz <= 1:
            return lst
        if sz % 2 == 1:
            mid = (sz - 1) // 2
            lim1 = mid
            start2 = mid + 1
        else:
            lim1 = sz // 2
            start2 = lim1
        result = lst[lim1:start2]
        seq1 = _predictable_shuffle(lst[:lim1], swap)
        seq2 = _predictable_shuffle(lst[start2:], not swap)
        pairs = zip(seq1, seq2)
        for (i, j) in pairs:
            if swap:
                i, j = j, i
            result.append(i)
            result.append(j)
            swap = not swap
        return result

    return _predictable_shuffle(list(seq), False)


def partition_files_by_size(file_paths: List[RichPath], bytes_per_part: int) -> List[List[RichPath]]:
    raw_graph_file_paths = predictable_shuffle(file_paths)  # type: List[RichPath]
    result = []
    current = []
    current_bytes = 0
    for raw_graph_file in raw_graph_file_paths:
        data_size = raw_graph_file.get_size()
        if current_bytes > 0 and current_bytes + data_size > bytes_per_part:
            result.append(current)
            current = []
            current_bytes = 0
        current.append(raw_graph_file)
        current_bytes += data_size

    # Don't forget to add the current part in:
    if current_bytes > 0:
        result.append(current)
    return result


def __parallel_queue_worker(worker_id: int,
                            job_queue: multiprocessing.Queue,
                            result_queue: multiprocessing.Queue,
                            worker_fn: Callable[[int, JobType], Iterable[ResultType]]):
    while True:
        job = job_queue.get()

        # "None" is the signal for last job, put that back in for other workers and stop:
        if job is None:
            job_queue.put(job)
            break

        for result in worker_fn(worker_id, job):
            result_queue.put(result)
    result_queue.put(None)


def run_jobs_in_parallel(all_jobs: List[JobType],
                         worker_fn: Callable[[int, JobType], Iterable[ResultType]],
                         received_result_callback: Callable[[ResultType], None],
                         finished_callback: Callable[[], None],
                         result_queue_size: int=100) -> None:
    """
    Runs jobs in parallel and uses callbacks to collect results.
    :param all_jobs: Job descriptions; one at a time will be parsed into worker_fn.
    :param worker_fn: Worker function receiving a job; many copies may run in parallel.
      Can yield results, which will be processed (one at a time) by received_result_callback.
    :param received_result_callback: Called when a result was produced by any worker. Only one will run at a time.
    :param finished_callback: Called when all jobs have been processed.
    """
    job_queue = multiprocessing.Queue(len(all_jobs) + 1)
    for job in all_jobs:
        job_queue.put(job)
    job_queue.put(None)  # Marker that we are done

    # This will hold the actual results:
    result_queue = multiprocessing.Queue(result_queue_size)

    # Create workers:
    num_workers = multiprocessing.cpu_count() - 1
    workers = [multiprocessing.Process(target=__parallel_queue_worker,
                                       args=(worker_id, job_queue, result_queue, worker_fn))
               for worker_id in range(num_workers)]
    for worker in workers:
        worker.start()

    num_workers_finished = 0
    while True:
        result = result_queue.get()
        if result is None:
            num_workers_finished += 1
            if num_workers_finished == len(workers):
                finished_callback()
                break
        else:
            received_result_callback(result)

    for worker in workers:
        worker.join()