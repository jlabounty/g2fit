import law 
import json
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import ast
import datetime

class GlobalParams(luigi.Config):
    date = luigi.DateParameter(default=datetime.date.today())
    campaignName = luigi.Parameter()
    outputDir = luigi.Parameter()