from collections import Iterator, MutableMapping
from functools import reduce
from operator import and_, or_
from typing import Callable, Dict, List, Tuple


class Iterate(MutableMapping):
    """
    A dict-like container to represent an iterate.

    Notes
    -----
    This class is needed to make sure that there is a key
    `iteration counter` in the dict
    """

    def __init__(self, *args, **kwargs):
        self.store = dict([('iteration counter', 0)])
        self.update(dict(*args, **kwargs))

    def __delitem__(self, key):
        del self.store[key]

    def __getitem__(self, key):
        return self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __repr__(self):
        return self.store.__repr__()


class Criterion:
    """
    Convergence criterion

    Attributes
    ----------

    threshold: float
        Comparison threshold
    comparison: Callable
        Operator to use for comparisons
    message: str
        Message to output when criterion is met
    """

    __slots__ = ['threshold', 'comparison', 'message']

    def __init__(self, threshold: float, comparison: Callable,
                 message: str) -> None:
        self.threshold = threshold
        self.comparison = comparison
        self.message = message.format(threshold=threshold)

    def compare(self, value) -> bool:
        """
        Compares value with preset threshold

        Parameters
        ----------
        value: float
            Value to be compared
        """
        return self.comparison(value, self.threshold)


class Stat:
    """
    Convergence statistics

    Attributes
    ----------

    header: str
        Label for the statistics, will be used in iterations output header
    fmt: str
        Format string for the values of the statistics, will be used in
        iterations report
    kind: str
        Kind of statistics, can be one of `report`, `failure`, `success`
    criterion: Criterion
        Convergence criterion for the statistics
    _satisfied: bool
        Whether the criterion was satisfied

    Notes
    -----
    We assume three types of statistics:
      - `report`. These are **not** checked for convergence, but merely reported during iterations.
        Example: the absolute value of the energy.
      - `failure`. These are checked for abnormal termination.
        Example: the number of iterations.
      - `success`. These are checked for normal termination.
        Example: the 2-norm of the residual vector.

    The `header` is the string you would like to see reported at the top of the
    iterations report, while `fmt` is the format string that should be used to
    report the value of the statistics.
    """
    __slots__ = ['header', 'fmt', 'kind', 'criterion', '_satisfied']

    def __init__(self,
                 header: str,
                 fmt: str,
                 kind: str,
                 criterion: Criterion = None) -> None:
        self.header = header
        self.fmt = fmt
        assert kind in [
            'failure', 'success', 'report'
        ], F"kind must be one of {', '.join(['failure', 'success', 'report'])}"
        self.kind = kind
        self.criterion = criterion
        self._satisfied = False

    @property
    def satisfied(self) -> bool:
        return self._satisfied

    @satisfied.setter
    def satisfied(self, value: bool) -> None:
        self._satisfied = value

    def __repr__(self):
        return F'{self.header:s} {self.kind:s} {self._satisfied}'


class IterativeSolver(Iterator):
    """
    Generic manager for iterative solvers.

    Attributes
    ----------
    _niterations: int
         Number of total iterations.
    _stepper: Callable
         Function encoding the update step in the iterative solver.
    _iterate: Iterate
         Current iterate.
    _stats: Dict[str, Stat]
         Dictionary with iterations statistics.
    _exception: Exception
         Exception to raise in case of failure.
    _checkpointer: Callable
         Function for checkpointing operations.
    _success_message: str
         Message to output in case of success.

    Notes
    -----
    This iterator encodes the following sequence of operations:
      1. Initialization.
      2. Failure checks.
      3. Iterate update. Return values from the stepper, if any, are ignored.
      4. Success checks.
      5. Iterations statistics report.
      6. Checkpointing operations.

    Apart from the initialization, which is carried out in the `__init__`
    method, all the other operations are in the `__next__` method.
    """
    __slots__ = [
        '_niterations', '_stepper', '_iterate', '_stats', '_exception',
        '_checkpointer', '_success_message'
    ]

    def __init__(self, stepper: Callable, start_guess: Iterate, stats: Dict,
                 exception, checkpointer: Callable) -> None:
        # Save number of iterations so far in global counter
        self._niterations = start_guess['iteration counter']
        self._stepper = stepper
        self._iterate = start_guess
        # Reset local counter, if necessary
        if start_guess['iteration counter'] != 0:
            self._iterate['iteration counter'] = 0
        # Generate dict of dicts, with kind of criteria as key
        self._stats = sort_stats_by_kind(stats)
        self._exception = exception
        self._checkpointer = checkpointer

        # Collate messages for success criteria
        self._success_message = '\n'.join(
            s.criterion.message for s in self._stats['success'].values())
        # Print iterations header
        print(self._header())
        # Print starting iterations statistics
        print(self._stat_line())

    @property
    def niterations(self) -> int:
        return self._niterations

    @property
    def iterate(self) -> Iterate:
        return self._iterate

    def _header(self) -> str:
        """
        Generate header for iterations report
        """
        how_many_stats = 0
        header = ''
        for v in self._stats.values():
            how_many_stats += len(v)
            header += ''.join(F'{x.header:^20s}' for x in v.values())
        header += F'\n{"=" * 20 * how_many_stats}'
        return header

    def _stat_line(self) -> str:
        """
        Generate line in iterations report
        """
        stat_line = ''
        for k, v in self._stats.items():
            fmt = ''.join(
                F"{{{x:s}:^ 20{y.fmt.strip('{:}'):s}}}{'*' if y.satisfied else ''}"
                for x, y in v.items())
            stat_line += fmt.format_map(
                {x: self._iterate[x]
                 for x in self._stats[k].keys()})
        return stat_line

    def __next__(self):
        try:
            # Check for failure
            k, failed = check_failure(self._stats['failure'], self._iterate)
            # Raise exception with correct message
            if failed:
                raise self._exception(
                    self._stats['failure'][k].criterion.message)
            else:
                _ = self._stepper(self._iterate)
            # Update global iterations counter
            self._niterations += 1
            # Check for success
            succeeded = check_success(self._stats['success'], self._iterate)
            # Print iterations statistics
            print(self._stat_line())
            if succeeded:
                raise StopIteration
        except self._exception:
            raise
        finally:
            # Clean up/checkpoint actions after each iteration
            self._checkpointer(self._iterate)
            if succeeded:
                print(self._success_message)


def sort_stats_by_kind(stats: Dict) -> Dict:
    """
    Generate dict-of-dicts for statistics.

    Returns
    -------
    Dict[str, Dict]
        A dict-of-dicts, where the key is the kind of the statistics in each dict.
    """
    failure = {}
    report = {}
    success = {}
    for k, v in stats.items():
        if v.kind == 'failure':
            failure[k] = v
        elif v.kind == 'success':
            success[k] = v
        else:
            report[k] = v
    return {'failure': failure, 'report': report, 'success': success}


def check_termination(combinator: Callable, criteria: Dict, value: Iterate,
                      initializer: bool) -> Tuple[str, bool]:
    """
    Checks termination

    Parameters
    ----------
    combinator: Callable
         Logic to apply to check termination
    criteria: Dict
         Dictionary of criteria to be checked
    value: Iterate
         Current iterate
    initializer: bool
         Base value of the comparison

    Returns
    -------
    Tuple[str, bool]
         A pair containing which criterion flipped `initializer` to `True`

    Warning
    -------
    This function is general, you should use `check_failure` and `check_success`

    Notes
    -----
    We want to decide, given one iterate and a list of criteria, whether these
    are met or not. In addition, we also want to know which ones where met.
    We classified criteria as "failure" and "success":
      - When _any_ of the "failure" criteria is met, we terminate iterations
        with an error message. Comparison predicates need to be composed
        _disjunctively_. `combinator` is thus `operator.or_`, while
        `initializer` is `False`. If multiple failing criteria are given, we
        want to know which one caused `initializer` to flip from `False` to
        `True`.
      - When _all_ of the "success" criteria are met, we terminate iterations
        with a success message. Comparison predicates need to be composed
        _conjunctively_. `combinator` is thus `operator.and_`, while
        `initializer` is `True`. Additionally, we want to know which criteria
        were satisfied.

    Due to the last requirement for "success" criteria, we cannot decide
    lazily, i.e. exiting as soon as the `initializer` flips its value, neither
    can we use just one loop.
    Hence:
      1. A first loop over the `criteria` dictionary will execute the `compare`
         method of the `criterion` attribute and correspondingly set the
        `satisfied` attribute.
      2. A second loop, in the form of a reduction, combines the `satisfied`
         attributes as prescribed by the `combinator` and `initializer`
         parameters.
    """

    terminate = ('', initializer)
    for k, v in criteria.items():
        v.satisfied = v.criterion.compare(value[k])
    return reduce(lambda x, y: (y[0], combinator(x[1], y[1].satisfied)),
                  criteria.items(), terminate)


def check_failure(criteria: Dict, value: Iterate) -> Tuple[str, bool]:
    """
    Specialization of `check_termination` for failure conditions

    Parameters
    ----------
    criteria: Dict
         Dictionary of criteria to be checked
    value: Iterate
         Current iterate

    Returns
    -------
    Tuple[str, bool]
         A pair containing which criterion flipped `initializer` to `True`
    """
    return check_termination(or_, criteria, value, False)


def check_success(criteria: Dict, value: Iterate) -> bool:
    """
    Specialization of `check_termination` for success conditions

    Parameters
    ----------
    criteria: Dict
         Dictionary of criteria to be checked
    value: Iterate
         Current iterate

    Returns
    -------
    bool
         Whether computation succeeded or not
    """
    return check_termination(and_, criteria, value, True)[1]
