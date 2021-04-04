"""Profiling tools
"""

import cProfile
import pstats
import time
import subprocess


class Profiler:
    """A wrapper around cProfile
    """

    def __init__(self, config):
        self.profiler = cProfile.Profile()
        self.has_run = False
        self.config = config
        self.tik = 0
        self.tok = 0

    def start(self):
        """start
        """
        self.tik = time.time()
        if self.config.profile:
            self.has_run = True
            self.profiler.enable()

    def stop(self):
        """stop
        """
        self.tok = time.time()
        if not self.has_run:
            return
        self.profiler.disable()
        self.profiler.dump_stats(self.config.base_file_name + 'profile.dump')

    def print_results(self):
        """print results
        """
        print("Total time used: {}".format(self.tok - self.tik))
        print("In average: {:0.2f} iterations / second".format(
            self.config.max_iteration / (self.tok - self.tik)))
        if not self.config.profile:
            return
        stats: pstats.Stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumtime')
        stats.print_stats()
        self.save_call_graph()

    def save_call_graph(self):
        """save the call graph
        """
        subprocess.run(
            [
                'gprof2dot -f pstats {} | dot -Tpng -o {}'.format(
                    self.config.base_file_name + 'profile.dump',
                    self.config.base_file_name + 'call-graph.png')
            ],
            shell=True,
            check=True)
