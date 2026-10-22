"""
Class that contains the cMaBoSS simulation of a BNet model.
"""
from .cmabosssimulation import CMaBoSSSimulation

class BNetSimulation(CMaBoSSSimulation):

    def __init__(self, bnet=None, cfgs=None, cmaboss=None, cmaboss_sim=None):

        if cmaboss is None or cmaboss_sim is None:
            self.bnet = bnet
            self.cfgs = cfgs

            self.nb_nodes = self.count_nodes()
            cmaboss = self.get_cmaboss()

            if self.cfgs is None:
                cmaboss_sim = cmaboss.MaBoSSSim(self.bnet)
            elif isinstance(self.cfgs, str):
                cmaboss_sim = cmaboss.MaBoSSSim(self.bnet, self.cfgs)
            else:
                cmaboss_sim = cmaboss.MaBoSSSim(self.bnet, configs=list(self.cfgs))

        CMaBoSSSimulation.__init__(self, cmaboss=cmaboss, cmaboss_sim=cmaboss_sim)

    def count_nodes(self):

        res = 0
        with open(self.bnet, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0 and "," in line and line.split(',')[0].strip() != "targets":
                    res += 1
        return res


__all__ = ["BNetSimulation"]
