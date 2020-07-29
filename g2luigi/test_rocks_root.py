import luigi
from root_to_boost import *
import logging
import os
from rocks_submission import SGEJobTask
from rocks_task import *
from global_params import *

if __name__ == "__main__":

    tasks = [ 
        # rocksWorkflowTest( ilow=1, iHigh=12, outDir="/home/labounty/luigi", dont_remove_tmp_dir=True),
        rootToBoost(dont_remove_tmp_dir=True, outDir="/home/labounty/luigi")
     ]
    luigi.build(tasks, local_scheduler=False, workers=1)