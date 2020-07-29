import json
import law
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import ast
import time
import numpy as np

import logging
import os
from rocks_submission import SGEJobTask

from global_params import GlobalParams
from start_time_scan import *

logger = logging.getLogger('luigi-interface')

class rocksTest(SGEJobTask):

    i = luigi.Parameter()
    outDir = luigi.Parameter()

    def work(self):
        logger.info('Running test job...')
        data = {"i":self.i}
        time.sleep(np.random.randint(3,10))
        # self.output().dump(data, formatter="json")
        self._smart_dump(data, formatter="json")

    def output(self):
        return law.LocalFileTarget(os.path.join(self.outDir, 'testfile_' + str(self.i)))


class rocksWorkflowTest(SGEJobTask):

    ilow = luigi.IntParameter()
    iHigh = luigi.IntParameter()
    outDir= luigi.Parameter() 

    def requires(self):
        for i in range(self.ilow, self.iHigh):
            yield rocksTest(i=str(i), n_cpu=2, run_locally=False, outDir=self.outDir, 
                job_name_format="TestLuigi", job_name="Test"+str(i), shared_tmp_dir="/home/labounty/", 
                dont_remove_tmp_dir=True, 
                parallel_env="smp")

    def work(self):
        inputs = []
        for inputi in self.input():
            print(inputi)
            inputs.append(inputi.path)
        data = {"inputs":inputs}    
        # self.output().dump(data, formatter="json")
        self._smart_dump(data, formatter="json")
        
    def output(self):
        return law.LocalFileTarget(os.path.join(self.outDir, 'testfile_summary'))

if __name__ == '__main__':
    # tasks = [ rocksTest(i=str(i), n_cpu=2, run_locally=False, outDir="/home/labounty/luigi", 
    #             job_name_format="TestLuigi", job_name="Test"+str(i), shared_tmp_dir="/home/labounty/", 
    #             dont_remove_tmp_dir=True, 
    #             parallel_env="smp" ) for i in range(6,9)]

    tasks = [ rocksWorkflowTest(ilow=1, iHigh=12, outDir="/home/labounty/luigi", dont_remove_tmp_dir=True) ]
    
    luigi.build(tasks, local_scheduler=False, workers=30)