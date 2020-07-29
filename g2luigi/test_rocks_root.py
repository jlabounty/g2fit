import luigi
from root_to_boost import *
import logging
import os
from rocks_submission import SGEJobTask
from rocks_task import *
from global_params import *
from g2_fit import *

if __name__ == "__main__":

    config = luigi.configuration.get_config()
    config.read('/home/labounty/github/g2fit/g2luigi/config.cfg')

    tasks = [ 
        # rocksWorkflowTest( ilow=1, iHigh=12, outDir="/home/labounty/luigi", dont_remove_tmp_dir=True),
        # rootToBoost(dont_remove_tmp_dir=True, outDir="/home/labounty/luigi"),
        g2FitHistAxisCuts(dont_remove_tmp_dir=True, outDir="/home/labounty/luigi", axisCuts=str([["y",1700,3200]]))
     ]
    luigi.build(tasks, local_scheduler=False, workers=1 )