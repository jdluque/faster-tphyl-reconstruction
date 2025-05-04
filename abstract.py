import copy

import pybnb


class BoundingAlgAbstract:
    def __init__(self):
        self.matrix = None
        self._extra_info = None
        self._extraInfo = {}
        self._times = {}
        self.na_support = False
        self.num_lower_bounds = 0
        # Stats about computer lower bounds
        self.lb_min_time = float("inf")
        self.lb_max_time = float("-inf")
        self.lb_time_total = 0
        pass

    def reset(self, matrix):
        raise NotImplementedError("The method not implemented")

    def get_bound(self, delta):
        """
        This bound should include the flips done so far too
        delta: a sparse matrix with fliped ones
        """
        raise NotImplementedError("The method not implemented")

    def get_name(self):
        return type(self).__name__

    def get_state(self):
        return None

    def set_state(self, state):
        assert state is None
        pass

    def get_extra_info(self):
        """
        Some bounding algorithms can provide extra information after calling bounding.
        E.g.,
        return {"icf":True, "bestPair":(a,b)}
        """
        return copy.copy(self._extraInfo)

    def get_priority(self, till_here, this_step, after_here, icf=False):
        return -after_here

    def get_times(self):
        return self._times

    def get_init_node(self) -> None | pybnb.Node:
        return None
