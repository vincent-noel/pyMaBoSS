from __future__ import print_function

from .result import EnsembleResult
from ..simulation import _default_parameter_list
from ..gsparser import load, _read_cfg
import os
import tempfile
import subprocess
import sys
from random import random
import shutil
import collections
import math

from colomoto import minibn

class Ensemble(object):

    def __init__(self, path, cfg_filename=None, individual_istates=collections.OrderedDict(), individual_mutations=collections.OrderedDict(), models=None, *args, **kwargs):

        self.models_path = path
        self.param = _default_parameter_list
        self.param["use_physrandgen"] = 0

        if models is None:
            self.models_files = [
                os.path.join(self.models_path, filename)
                for filename in os.listdir(self.models_path)
                if filename.endswith(".bnet") or filename.endswith(".bnd")
            ]
        else:
            self.models_files = [
                os.path.join(self.models_path, filename)
                for filename in models
            ]

        self.minibns = None
        self.miniensemble = None

        self.nodes = []
        self.read_nodes(self.models_files[0])

        self.variables = collections.OrderedDict()
        self.istates = collections.OrderedDict()
        self.individual_istates = individual_istates
        self.individual_mutations = individual_mutations
        self.individual_cfgs = collections.OrderedDict()
        self.outputs = collections.OrderedDict()
        self.mutations = collections.OrderedDict()
        self.individual_results = False
        self.random_sampling = False
        self.palette = {}

        if cfg_filename is not None:
            with open(cfg_filename, 'r') as cfg_file:
                variables, params, internals, istates, refstates = _read_cfg(cfg_file.read())
                self.outputs.update({node:not value for node, value in internals.items()})
                self.istates = istates
                self.param = params

        for p in kwargs:
            if p in self.param:
                # That will keep the existing ones if unchanged
                self.param[p] = kwargs[p]
            
            elif p == "istate":
                # Here we reset it in case we have bound states
                self.istates.update({node: {0:1, 1:0} for node in self.nodes})
                self.istates.update({node: {0:val0, 1:val1} for node, (val0, val1) in kwargs[p].items()})
        
            elif p == "outputs":
                # Here we configure for all nodes, aka we reset it
                self.outputs.update({node: node in kwargs[p] for node in self.nodes})

            elif p == "mutations":
                # Here we update the existing values
                self.mutations.update(kwargs[p])

            elif p == "individual_results":
                self.individual_results = kwargs[p]

            elif p == "random_sampling":
                self.random_sampling = kwargs[p]

    def copy(self):
        ensemble = Ensemble(self.models_path)
        ensemble.param = self.param.copy()
        ensemble.variables = self.variables.copy()
        ensemble.istates = self.istates.copy()
        ensemble.outputs = self.outputs.copy()
        ensemble.individual_results = self.individual_results
        ensemble.random_sampling = self.random_sampling
        ensemble.palette = self.palette.copy()
        ensemble.individual_istates = self.individual_istates.copy()
        ensemble.models_files = self.models_files.copy()
        ensemble.nodes = self.nodes.copy()
        return ensemble

    def get_maboss_cmd(self):
        maboss_cmd = "MaBoSS"

        l = len(self.nodes)
        assert l <= 1024, "MaBoSS is not compatible with models with more than 1024 nodes"
        if l <= 64:
            pass
        elif l <= 128:
            maboss_cmd = "MaBoSS_128n"
        elif l <= 256:
            maboss_cmd = "MaBoSS_256n"
        elif l <= 516:
            maboss_cmd = "MaBoSS_512n"
        elif l <= 1024:
            maboss_cmd = "MaBoSS_1024n"

        return maboss_cmd

    def run(self, workdir=None, overwrite=False, prefix="res"):
        return EnsembleResult(self, workdir, overwrite, prefix)
        # return EnsembleResult(self.models_files, self._cfg, "res", self.individual_results, self.random_sampling)

    def set_models_path(self, path):
        self.models_path = path
        self.models_files = [
            os.path.join(self.models_path, filename) 
            for filename in os.listdir(self.models_path)
            if filename.endswith(".bnet") or filename.endswith(".bnd")
        ]

    def mutate(self, node, state):
        """
        Trigger or untrigger mutation for a node.

        :param node: The identifier of the node to be modified
        :type node: :py:str:
        :param str State:

            * ``'ON'`` (always up)
            * ``'OFF'`` (always down)
            * ``'WT'`` (mutable but with normal behaviour)


        The node will appear as a mutable node in the bnd file.
        This means that its rate will be of the form:

        ``rate_up = $LowNode ? 0 :($HighNode ? 1: (@logic ? rt_up : 0))``

        If the node is already mutable, this method will simply set $HighNode
        and $LowNode accordingly to the desired mutation.
        """
        if node not in self.nodes:
            print("Error, unknown node %s" % node, file=sys.stderr)
            return

        self.mutations.update({node: state})

    def set_istate(self, nodes, probDict, warnings=True):
        """
        Change the inital states probability of one or several nodes.

        :param nodes: the node(s) whose initial states are to be modified
        :type nodes: a :py:class:`Node` or a list or tuple of :py:class:`Node`
        :param dict probDict: the probability distribution of intial states

        If nodes is a Node object or a singleton, probDict must be a probability
        distribution over {0, 1}, it can be expressed by a list [P(0), P(1)] or a
        dictionary: {0: P(0), 1: P(1)}.

        If nodes is a tuple or a list of several Node objects, the Node object 
        will be bound, and probDict must be a probability distribution over a part
        of {0, 1}^n. It must be expressed in the form of a dictionary
        {(b1, ..., bn): P(b1,..,bn),...}. States that do not appear in the 
        dictionary will be considered to be impossible. If a state has a 0 probability of
        being an intial state but might be reached later, it must explicitly appear 
        as a key in probDict.
        
        **Example**
        
        >>> my_network.set_istate('node1', [0.3, 0.7]) # node1 will have a probability of 0.7 of being up
        >>> my_network.set_istate(['node1', 'node2'], {(0, 0): 0.4, (1, 0): 0.6, (0, 1): 0}) # node1 and node2 can never be both up because (1, 1) is not in the dictionary
        """ 
        self.istates.update({nodes: {i:val for i, val in enumerate(probDict)}})



    def set_outputs(self, outputs):
        self.outputs.update({node: node in outputs for node in self.nodes})

    def get_mini_bns(self):
        if self.minibns is None:
            self.minibns = []
            for model_file in self.models_files:
                assert model_file.lower().endswith(".bnet"), \
                        "Only .bnet files are supported as input"
                bn = minibn.BooleanNetwork.load(model_file)
                self.minibns.append(bn)

        return self.minibns

    def get_mini_ensemble(self, cluster=None):
        
        minibns = self.get_mini_bns()

        if cluster is not None:
            minibns = [minibn for i, minibn in enumerate(minibns) if i in cluster]

        miniensemble = {node:set() for node in minibns[0].keys()}
        for minibn in minibns:
            for node in minibn.keys():
                miniensemble[node].add(minibn[node])

        return miniensemble

    def print_ensemble_stats(self, cluster=None):

        mini_ensemble = self.get_mini_ensemble(cluster)
        nodes_selection_rate = {}

        for node, list_rules in mini_ensemble.items():
            
            nb_rules = len(list_rules)
            nb_nodes = len(next(iter(list_rules)).symbols)
            
            nodes_selection_rate.update({
                node: (nb_rules/(pow(2, pow(2, nb_nodes-1)-1))) if nb_nodes > 0 else math.nan
            })
            print("%s : %g (%d, %d)" % (node, nodes_selection_rate[node] , nb_rules, nb_nodes))

    def get_nodes_selection_rate(self, cluster=None):
        mini_ensemble = self.get_mini_ensemble(cluster)
        nodes_selection_rate = {}
        
        for node, list_rules in mini_ensemble.items():
            nb_rules = len(list_rules)
            nb_nodes = len(next(iter(list_rules)).symbols)

            nodes_selection_rate.update({
                node: (nb_rules/(pow(2, pow(2, nb_nodes-1)-1))) if nb_nodes > 0 else math.nan
            })

        
        return collections.OrderedDict(sorted(nodes_selection_rate.items(), key=lambda kv: kv[1]))

    def compare_nodes_selection_rate(self, cluster):
        rate_wt = self.get_nodes_selection_rate()
        rate_cluster = self.get_nodes_selection_rate(cluster)

        res = {node: rate_cluster[node]/value for node, value in rate_wt.items()}
        return collections.OrderedDict(sorted(res.items(), key=lambda kv: kv[1]))

    def read_nodes(self, filename):
        if os.path.splitext(filename)[1] == ".bnet":
            self.read_bnet_nodes(filename)
        elif os.path.splitext(filename)[1] == ".bnd":
            self.read_maboss_nodes(filename)
        else:
            print("Unknown model type")

    def read_maboss_nodes(self, filename):
        model = load(filename, self._cfg)
        self.nodes = list(model.network.keys())

    def read_bnet_nodes(self, filename):
        self.nodes = []
        with open(filename, 'r') as bnet_model:
            for line in bnet_model:
                self.nodes.append(line.split(",")[0].strip())

    def str_cfg(self, individual=None):
        res = ""
        for var, value in self.variables.items():
            res += "%s = %s;\n" % (var, value)

        for param, value in self.param.items():
            res += "%s = %s;\n" % (param, value)

        if individual is None:
            defined = []
            for node, istate in self.istates.items():
                if isinstance(node, tuple):
                    defined += list(node)
                    res += "["
                    for i, t_node in enumerate(node):
                        res += t_node
                        if i < len(node)-1:
                            res += ", "
                    res += "].istate = "

                    for i, (state, value) in enumerate(istate.items()):
                        res += "%f %s" % (value, str(list(state)).replace(" ", ""))
                        if i < len(istate)-1:
                            res += ", "
                    res += ";\n"
                else:
                    defined.append(node)
                    res += "[%s].istate = %.1f[0], %.1f[1];\n" % (node.replace("-", "_"), istate[0], istate[1])
                    
            for node in list(set(self.nodes).difference(set(defined))):
                res += "[%s].istate = %.1f[0], %.1f[1];\n" % (node.replace("-", "_"), 1.0, 0.0)
 
        else:

            for node, istate in self.individual_istates[individual].items():
                if isinstance(node, tuple):
                    res += "["
                    for i, t_node in enumerate(node):
                        res += t_node
                        if i < len(node) - 1:
                            res += ", "

                    res += "].istate = "

                    for i, (nodes, value) in enumerate(istate.items()):
                        res += "%f %s" % (value, str(list(nodes)))
                        if i < len(istate) - 1:
                            res += ", "

                    res += ";\n"

                else:
                    if istate not in [0, 1]:
                        istate = 0

                    res += "[%s].istate = %.1f[0], %.1f[1];\n" % (node.replace("-", "_"), 1 - istate, istate)

        for node in self.nodes:
            if len(self.outputs) == 0 or (node in self.outputs.keys() and self.outputs[node]):
                res += "%s.is_internal = FALSE;\n" % node.replace("-", "_")
            else:
                res += "%s.is_internal = TRUE;\n" % node.replace("-", "_")

        return res

    def print_cfg(self):
        print(self.str_cfg())

    def write_cfg(self, path, filename):
        if os.path.exists(os.path.join(path, "models")):
            shutil.rmtree(os.path.join(path, "models"))

        os.mkdir(os.path.join(path, "models"))

        if len(self.individual_istates) == 0:
            with open(os.path.join(path, filename), 'w+') as cfg_file:
                cfg_file.write(self.str_cfg())
        else:
            self.individual_cfgs = collections.OrderedDict()
            for individual in self.individual_istates.keys():
                t_filename = "%s.cfg" % ".".join(individual.split(".")[0:-1])
                self.individual_cfgs.update({individual: t_filename})
                with open(os.path.join(path, "models", t_filename), 'w+') as cfg_file:
                    cfg_file.write(self.str_cfg(individual))

    def write_models(self, path):
        new_models = []

        for model in self.models_files:
            if os.path.splitext(model)[1] == ".bnet":
                new_models.append(self.mutate_bnet(model, path))
        self.models_files = new_models

    def mutate_bnet(self, model_path, path):
        new_path = os.path.join(path, "models", os.path.basename(model_path))

        with open(model_path, 'r') as model_file, open(new_path, 'w+') as new_model_file:
            for line in model_file:
                var, formula = line.split(",")
                var = var.strip()

                while "-" in var:
                    var = var.replace("-", "_")

                if var in self.mutations.keys():
                    if self.mutations[var] == 'ON':
                        new_model_file.write("%s, %d\n" % (var, 1))

                    elif self.mutations[var] == 'OFF':
                        new_model_file.write("%s, %d\n" % (var, 0))

                    else:
                        print("Unknown mutation %s for node %s. Ignored" % (self.mutations[var], var))
                else:
                    formula = formula.strip()

                    while "-" in formula:
                        formula = formula.replace("-", "_")

                    new_model_file.write("%s, %s\n" % (var, formula))

        return new_path

def state2bool(state, nodes):
    res = [0] * len(nodes)

    if state == "<nil>":
        return tuple(res)

    for t_node in state.split(" -- "):
        res[nodes.index(t_node)] = 1

    return tuple(res)
