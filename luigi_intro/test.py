import law 
import json
import luigi
import os

class firstLuigi(law.Task):

    n = luigi.IntParameter()  # no default!
    
    def output(self):
        return law.LocalFileTarget(
            f"data/n{self.n}.json")
    
    def run(self):        
        # method 1: the verbose way
        output = self.output()
        output.parent.touch()  # creates the data/ dir

        # define data to save
        # note: self.n is the value of the "n" parameter
        # and != self.__class__.n (parameter instance!)
        data = {"in": self.n, "out": self.n * 2}

        # pythonic way to save data
        with open(output.path, "w") as f:
            json.dump(data, f, indent=4)