from __future__ import print_function

from ..result import BaseResult
import os
import tempfile
import subprocess
import sys

class EnsembleResult(BaseResult):

    def __init__(self, path, cfg_filename):
        BaseResult.__init__(self)

        self.models_path = path
        self._cfg = cfg_filename

        self._path = tempfile.mkdtemp()
        self.models_files = [os.path.join(self.models_path, file) for file in os.listdir(self.models_path)]
        
        maboss_cmd = "MaBoSS"
        cmd_line = [maboss_cmd, "-c", self._cfg,'--ensemble',  "-o", self._path+'/res'] + self.models_files
        
        res = subprocess.Popen(
            cmd_line,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        std_out, std_err = res.communicate()
        self._err = res.returncode
        if self._err != 0:
            print("Error, MaBoSS returned non 0 value", file=sys.stderr)
            print(std_err.decode())
        
        if len(std_out.decode()) > 0:
            print(std_out.decode())

             
    def get_fp_file(self):
        return os.path.join(self._path, "res_fp.csv")

    def get_probtraj_file(self):
        return os.path.join(self._path, "res_probtraj.csv")
        
        