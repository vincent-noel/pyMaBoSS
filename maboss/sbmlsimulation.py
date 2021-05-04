"""
Class that contains the cMaBoSS simulation.
"""
from .sbmlcmabossresult import SBMLCMaBoSSResult

class SBMLSSimulation(object):

    def __init__(self, sbml, cfgs):

        self.sbml = sbml
        self.cfgs = cfgs

        self.nb_nodes = self.count_nodes()
        self.cmaboss = self.get_cmaboss()

        self.cmaboss_sim = self.cmaboss.MaBoSSSim(self.sbml, self.cfgs)

        # self.cmaboss_net = self.cmaboss.MaBoSSNet(self.bnd)
        # self.cmaboss_cfg = self.cmaboss.MaBoSSCfg(self.cmaboss_net, self.cfgs)
        
    def count_nodes(self):
        
        res = 0
        with open(self.sbml, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0 and "</qual:qualitativespecies>" in line.lower():
                    res += 1
        return res


    def get_cmaboss(self):

        assert self.nb_nodes <= 1024, "Models with more than 1024 nodes are not compatible with this version of MaBoSS"

        if self.nb_nodes <= 64:
            return __import__("cmaboss")
        elif self.nb_nodes <= 128:
            return __import__("cmaboss_128n")
        elif self.nb_nodes <= 256:
            return __import__("cmaboss_256n")
        elif self.nb_nodes <= 512:
            return __import__("cmaboss_512n")
        else:
            return __import__("cmaboss_1024n")

    def get_logical_rules(self):

        out = self.cmaboss_sim.get_logical_rules()    
        rules = {}
        for line in out.split("\n"):
            if ":" in line:
                node, rule = line.split(" : ", 1)
                rules.update({node.strip(): rule.strip()})

        return rules


    def run(self, only_final_state=False):
        return SBMLCMaBoSSResult(self, only_final_state)


__all__ = ["SBMLSSimulation"]
