from collections import Iterator, OrderedDict
from typing import Callable, Dict, List, Tuple


class Criterion:
    def __init__(self, threshold: float, comparison: Callable,
                 message: str) -> None:
        self.threshold = threshold
        self.comparison = comparison
        self.message = message.format(threshold=threshold)

    def compare(self, value):
        return self.comparison(value, self.threshold)


class Stat:
    def __init__(self,
                 header: str,
                 fmt: str,
                 kind: str,
                 criterion: Criterion = None) -> None:
        self.header = header
        self.fmt = fmt
        if kind not in ['failure', 'success', 'report']:
            raise RuntimeError('\'kind\' must be one of {}'.format(
                ['failure', 'success', 'report']))
        self.kind = kind
        self.criterion = criterion


def check_termination(composition: str, criteria: List,
                      value: Dict) -> Tuple[int, bool]:
    """
    composition: str
      Denotes the composition of predicates, only disjucntive ('any') or conjuctive ('all') are allowed
    criteria:
      List of criteria conforming to the Criterion interface
    value:
      Current iterate
    """

    idx = 0
    terminate = False
    if composition == 'any':
        idx, terminate = any_with_index(
            map(lambda c: c.compare(value), criteria))
    elif composition == 'all':
        terminate = all(map(lambda c: c.compare(value), criteria))
    else:
        raise ValueError(
            'Composition of predicates \'{:s}\' not recognized.\n Only \'any\' and \'all\' are allowed.'.
            format(composition))
    return idx, terminate


def any_with_index(iterable):
    for idx, element in enumerate(iterable):
        if element:
            return idx, True
    return idx, False


def check_termination(what: str, criteria: Dict,
                      value: Dict) -> Tuple[int, bool]:
    """
    what: str
      'failure' (disjuncton) or 'success' (conjunction)
    criteria:
    value:
      Current iterate
    """

    terminate = False
    if what == 'failure':
        conds = {k: v for k, v in criteria.items() if v.kind == 'failure'}
        terminate = any(
            (conds[k].criterion.compare(value[k]) for k in conds.keys()))
    elif what == 'success':
        conds = {k: v for k, v in criteria.items() if v.kind == 'success'}
        terminate = all(
            (conds[k].criterion.compare(value[k]) for k in conds.keys()))
    else:
        raise ValueError(
            'Unknown check \'{:s}\'.\n Only \'failure\' and \'success\' are allowed.'.
            format(what))
    return terminate


class IterativeSolver(Iterator):
    def __init__(self, stepper: Callable, start_guess: Dict, stats: Dict,
                 exception) -> None:
        self._niterations = start_guess['iteration counter']
        self._stepper = stepper
        self._iterate = start_guess
        self._stats = self._sort_stats_by_kind(stats)
        self._success_message = '\n'.join(s.criterion.message
                                          for s in self._stats.values()
                                          if s.kind == 'success')
        self._exception = exception
        print(self._header())

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
        header = ''.join(
            '{:^20s}'.format(v.header) for v in self._stats.values())
        header += '\n{:s}'.format('=' * 20 * nheaders)
        return header

    def _stat_line(self) -> str:
        stats = {k: self._iterate[k] for k, _ in self._stats.items()}
        # Prepare format for stat line by stringing together the fmt strings
        # See here for explanation of format specification: https://pyformat.info/#getitem_and_getattr
        fmts = ''.join('{{stats[{:s}]:^20{:s}}}'.format(k, v.fmt.strip('{:}'))
                       for k, v in self._stats.items())
        return fmts.format(stats=stats)

    def __next__(self):
        try:
            failed = check_termination('failure', self._stats, self._iterate)
            if failed:
                raise self._exception  #(self._stats[idx].criterion.message)
            else:
                self._iterate = self._stepper(self._iterate)
            # Update iterations statistics
            print(self._stat_line())
            # Update global iterations counter
            self._niterations += 1
            # Check for success
            succeeded = check_termination('success', self._stats,
                                          self._iterate)
            if succeeded:
                print(self._success_message)
                raise StopIteration
        except self._exception:
            raise
        finally:
            # Clean up/checkpoint actions after each iteration
            pass
