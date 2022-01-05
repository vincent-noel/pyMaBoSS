def simulate_single_mutants(model, list_nodes, sign="BOTH"):
    
    list_single_mutants = []
    if (sign == "BOTH" or sign == "ON"):
        list_single_mutants += [(node, "ON") for node in list_nodes]
    if (sign == "BOTH" or sign == "OFF"):
        list_single_mutants += [(node, "OFF") for node in list_nodes]
    
    res = {}
    for single_mutant in list_single_mutants:
        t_model = model.copy()
        t_model.mutate(*single_mutant)
        res.update({single_mutant: t_model.run()})
        
    return res
    
    
def simulate_double_mutants(model, list_nodes, sign="BOTH"):
    
    list_single_mutants = []
    if (sign == "BOTH" or sign == "ON"):
        list_single_mutants += [(node, "ON") for node in list_nodes]
    if (sign == "BOTH" or sign == "OFF"):
        list_single_mutants += [(node, "OFF") for node in list_nodes]
    
    list_double_mutants = [(a, b) for idx, a in enumerate(list_single_mutants) for b in list_single_mutants[idx + 1:] if a[0] != b[0]]

    res = {}
    for double_mutant in list_double_mutants:
        t_model = model.copy()
        t_model.mutate(*(double_mutant[0]))
        t_model.mutate(*(double_mutant[1]))
        res.update({double_mutant: t_model.run()})
        
    return res
    
def filter_sensititivy(results, state=None, node=None, minimum=None, maximum=None):
    ret_res = {}
    for (mutant, res) in results.items():
        
        if state is not None:
            t_res = res.get_last_states_probtraj()
            if state in t_res.columns:
                if minimum is not None and maximum is not None:
                    if t_res[state].values[0] > minimum and t_res[state].values[0] < maximum:
                        ret_res.update({mutant: res})
                elif minimum is not None and t_res[state].values[0] > minimum:
                    ret_res.update({mutant: res})
                elif maximum is not None and t_res[state].values[0] < maximum:
                    ret_res.update({mutant: res})
            elif maximum is not None and minimum is None:
                ret_res.update({mutant:res})

        elif node is not None:
            
            t_res = res.get_last_nodes_probtraj()

            if node in t_res.columns:
                if minimum is not None and maximum is not None:
                    if t_res[node].values[0] > minimum and t_res[node].values[0] < maximum:
                        ret_res.update({mutant: res})
                elif minimum is not None and t_res[node].values[0] > minimum:
                    ret_res.update({mutant: res})
                elif maximum is not None and t_res[node].values[0] < maximum:
                    ret_res.update({mutant: res})
            elif maximum is not None and minimum is None:
                ret_res.update({mutant: res})

    return ret_res
