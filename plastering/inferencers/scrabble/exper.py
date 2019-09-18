import sys
import logging
import pdb
import json
from collections import defaultdict

import arrow

from ir2tagsets_seq import Ir2Tagsets

ground_truth_types = {
    Ir2Tagsets: 'tagsets_dict'
}


class Exper(object):
    def __init__(self,
                 inferencer,
                 eval_functions,
                 configs={}):
        self.inferencer = inferencer
        self.eval_functions = eval_functions
        self.ground_truth_type = ground_truth_types[type(inferencer)]

        # configs
        if 'step_num' in configs:
            self.step_num = configs['step_num']
        else:
            self.step_num = 10

        if 'iter_num' in configs:
            self.iter_num = configs['iter_num']
        else:
            self.iter_num = 25

        if 'logfile' in configs:
            logfile = configs['logfile']
        else:
            logfile = 'logs/{0}_{1}.log'.format(type(self.inferencer).__name__,
                                                (arrow.get()))
        self.set_logger(logfile)
        self.history = []

    def set_logger(self, logfile=None):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s')

        # Console Handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

        # File Handler
        if logfile:
            fh = logging.FileHandler(logfile, mode='w+')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        self.logger = logger
        #return logger

    def evaluate(self, pred):
        truth_dict = getattr(self.inferencer, self.ground_truth_type)
        truth_dict = {srcid: truth_dict[srcid] for srcid in pred.keys()}
        res = {}
        for eval_type, eval_func in self.eval_functions.items():
            res[eval_type] = eval_func(pred, truth_dict)
        return res

    def _log(self, s):
        self.logger.info(s)

    def update_history(self, metrics):
        one_history = {
            'metrics': metrics,
            'num_samples': self.inferencer.get_learning_sample_nums(),
        }
        self.history.append(one_history)


    def run_exp(self):
        self.inferencer.update_model([])
        for i in range(0, self.iter_num):
            t0 = arrow.get()
            new_srcids = self.inferencer.select_informative_samples(
                self.step_num)
            print('pass')
            for srcid in new_srcids:
                if srcid in self.inferencer.learning_srcids:
                    print('WARNING: {0} is selected again.'.format(srcid))
            self.inferencer.update_model(new_srcids)
            pred = self.inferencer.predict(
                [srcid for srcid in self.inferencer.target_srcids
                 if srcid not in self.inferencer.learning_srcids])
            metrics = self.evaluate(pred)
            t1 = arrow.get()
            self.update_history(metrics)
            print('{0}th took: {1}'.format(i, t1 - t0))
            self._log('{0}th took: {1}'.format(i, t1 - t0))
            print('Metrics: {0}'.format(metrics))
            self._log('Metrics: {0}'.format(metrics))
        #proba = self.inferencer.predict_proba(target_srcids)
        with open('result/{0}_{1}.json'.format(type(self.inferencer).__name__,
                                               (arrow.get())), 'w') as fp:
            json.dump(self.history, fp, indent=2)

