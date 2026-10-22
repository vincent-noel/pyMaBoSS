"""
Class that contains the cMaBoSS simulation of a SBML model.
"""
from .cmabosssimulation import CMaBoSSSimulation

class SBMLSSimulation(CMaBoSSSimulation):

    def __init__(self, sbml=None, cfgs=None, use_sbml_names=False, cmaboss=None, cmaboss_sim=None):

        if cmaboss is None or cmaboss_sim is None:
            self.sbml = sbml
            self.cfgs = cfgs

            self.nb_nodes = self.count_nodes()
            cmaboss = self.get_cmaboss()

            if self.cfgs is None:
                cmaboss_sim = cmaboss.MaBoSSSim(self.sbml, use_sbml_names=use_sbml_names)
            elif isinstance(self.cfgs, str):
                cmaboss_sim = cmaboss.MaBoSSSim(self.sbml, self.cfgs, use_sbml_names=use_sbml_names)
            else:
                cmaboss_sim = cmaboss.MaBoSSSim(self.sbml, configs=list(self.cfgs), use_sbml_names=use_sbml_names)

        CMaBoSSSimulation.__init__(self, cmaboss=cmaboss, cmaboss_sim=cmaboss_sim)

    def count_nodes(self):

        res = 0
        with open(self.sbml, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0 and "<qual:qualitativespecies " in line.lower():
                    res += line.lower().count("<qual:qualitativespecies ")
        return res


__all__ = ["SBMLSSimulation"]
