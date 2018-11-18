from collections import Iterator, MutableMapping, OrderedDict
from functools import reduce
from operator import and_, or_
from typing import Callable, Dict, List, Tuple


class Iterate(MutableMapping):
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
    def __init__(self, threshold: float, comparison: Callable,
                 message: str) -> None:
        self.threshold = threshold
        self.comparison = comparison
        self.message = message.format(threshold=threshold)

    def compare(self, value) -> bool:
        return self.comparison(value, self.threshold)


class Stat:
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


def check_termination(combinator: Callable, criteria: Dict, value: Iterate,
                      initializer: bool) -> Tuple[str, bool]:
    """
    combinator: Callable
    criteria:
    value:
      Current iterate
    initializer: bool

    Notes
    -----
    We cannot decide with just one loop, since we want to check all convergence
    criteria.
    """

    terminate = ('', initializer)
    for k, v in criteria.items():
        v.satisfied = v.criterion.compare(value[k])
    return reduce(lambda x, y: (y[0], combinator(x[1], y[1].satisfied)),
                  criteria.items(), terminate)


class IterativeSolver(Iterator):
    def __init__(self, stepper: Callable, start_guess: Iterate, stats: Dict,
                 exception, checkpointer: Callable) -> None:
        # Save number of iterations so far in global counter
        self._niterations = start_guess['iteration counter']
        self._stepper = stepper
        self._iterate = start_guess
        # Reset local counter, if necessary
        if start_guess['iteration counter'] != 0:
            self._iterate['iteration counter'] = 0
        self._stats = self._sort_stats_by_kind(stats)
        self._exception = exception
        self._checkpointer = checkpointer

        self._success_message = '\n'.join(s.criterion.message
                                          for s in self._stats.values()
                                          if s.kind == 'success')
        self._failure = {
            k: v
            for k, v in self._stats.items() if v.kind == 'failure'
        }
        self._success = {
            k: v
            for k, v in self._stats.items() if v.kind == 'success'
        }
        # Print iterations header
        print(self._header())
        # Print starting iterations statistics
        print(self._stat_line())

    @property
    def niterations(self) -> int:
        return self._niterations

    @property
    def iterate(self) -> Dict:
        return self._iterate

    def _sort_stats_by_kind(self, stats: Dict) -> OrderedDict:
        """
        Sort stats dictionary in the failure, report, success order
        """
        tmp = OrderedDict(
            {k: v
             for k, v in stats.items() if v.kind == 'failure'})
        tmp.update({k: v for k, v in stats.items() if v.kind == 'report'})
        tmp.update({k: v for k, v in stats.items() if v.kind == 'success'})
        return tmp

    def _header(self) -> str:
        nheaders = len(self._stats)
        header = ''.join(F'{v.header:^20s}' for v in self._stats.values())
        header += F'\n{"=" * 20 * nheaders}'
        return header

    def _stat_line(self) -> str:
        fmts = ''.join(
            F"{{{k:s}:^ 20{v.fmt.strip('{:}'):s}}}{'*' if v.satisfied else ''}"
            for k, v in self._stats.items())
        return fmts.format_map(
            {k: self._iterate[k]
             for k in self._stats.keys()})

    def __next__(self):
        try:
            k, failed = check_termination(or_, self._failure, self._iterate,
                                          False)
            if failed:
                raise self._exception(self._failure[k].criterion.message)
            else:
                self._stepper(self._iterate)
            # Update global iterations counter
            self._niterations += 1
            # Check for success
            _, succeeded = check_termination(and_, self._success,
                                             self._iterate, True)
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
