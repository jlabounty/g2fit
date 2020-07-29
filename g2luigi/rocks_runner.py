# -*- coding: utf-8 -*-
#
# Copyright 2012-2015 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The SunGrid Engine runner

The main() function of this module will be executed on the
compute node by the submitted job. It accepts as a single
argument the shared temp folder containing the package archive
and pickled task to run, and carries out these steps:

- extract tarfile of package dependencies and place on the path
- unpickle SGETask instance created on the master node
- run SGETask.work()

On completion, SGETask on the master node will detect that
the job has left the queue, delete the temporary folder, and
return from SGETask.run()
"""

import os
import sys
import pickle
import logging
import tarfile
from pathlib import Path
import random 
import time


def _do_work_on_compute_node(work_dir, tarball=True):

    #get temporary directories for processing on rocks, stored in env variables
    tmp_dir = os.environ['TMPDIR']
    storage_dir = "/state/partition1"

    os.chdir(tmp_dir)
    files_to_copy = [work_dir+"/"+x for x  in os.listdir(work_dir)]
    _smart_copy(files_to_copy, tmp_dir)

    if tarball:
        # Extract the necessary dependencies
        # This can create a lot of I/O overhead when running many SGEJobTasks,
        # so is optional if the luigi project is accessible from the cluster node
        _extract_packages_archive(tmp_dir)

    # Open up the pickle file with the work to be done
    # os.chdir(work_dir)
    with open("job-instance.pickle", "rb") as f:
        job = pickle.load(f)

    # Do the work contained
    job.work()

def _smart_copy(from_paths, to_paths, queue_dir="/home/labounty/stupid_queue/"):
    '''
        Python implementation of the 'stupid queue' we have set up on rocks
        Bash implementation for reference
        while :
        do
            nfiles=$( ls -l ${queue_dir} | wc -l)
            echo "$nfiles in directory"

            if [ "$nfiles" -le "10" ]
            then
                echo Processing!
                touch ${queue_dir}${datestring}
                rsync -vz --progress  ${infilepath}/${infile} $ROOTSTORAGE/
                rsync -vz --progress  ${scriptPath}${script} $TMPDIR/
                rm -f ${queue_dir}${datestring}
                break
            fi
            echo "waiting..."
            sleep 10
        done 
    '''
    if type(from_paths) is not list:
        from_paths = [from_paths]
        to_paths = [to_paths]
    elif type(to_paths) is not list:
        to_paths = [to_paths for i in range(len(from_paths))]
    assert len(from_paths) == len(to_paths)
    print("Initiating smart copy using queue:", queue_dir)
    waiting_for_copy = True
    while( waiting_for_copy ):
        nfiles = len(os.listdir(queue_dir))
        if(nfiles < 10):
            thisFile = '%016x' % random.getrandbits(64)
            Path(queue_dir+thisFile).touch()
            for from_path, to_path in zip(from_paths, to_paths):
                print("Processing copy(ies) from", from_path, "to", to_path)
                os.system(f"rsync -vz --progress {from_path} {to_path}")
            Path(queue_dir+thisFile).unlink()
            waiting_for_copy = False
        else:
            print("waiting... (nfiles: ", nfiles,")")
            time.sleep(10)



def _extract_packages_archive(work_dir):
    package_file = os.path.join(work_dir, "packages.tar")
    if not os.path.exists(package_file):
        return

    curdir = os.path.abspath(os.curdir)

    os.chdir(work_dir)
    tar = tarfile.open(package_file)
    for tarinfo in tar:
        tar.extract(tarinfo)
    tar.close()
    if '' not in sys.path:
        sys.path.insert(0, '')

    os.chdir(curdir)


def main(args=sys.argv):
    """Run the work() method from the class instance in the file "job-instance.pickle".
    """
    try:
        tarball = "--no-tarball" not in args
        # Set up logging.
        logging.basicConfig(level=logging.WARN)
        work_dir = args[1]
        assert os.path.exists(work_dir), "First argument to sge_runner.py must be a directory that exists"
        project_dir = args[2]
        sys.path.append(project_dir)
        _do_work_on_compute_node(work_dir, tarball)
    except Exception as e:
        # Dump encoded data that we will try to fetch using mechanize
        print(e)
        raise


if __name__ == '__main__':
    main()