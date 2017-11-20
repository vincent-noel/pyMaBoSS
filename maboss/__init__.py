from .network import Node, Network
from .simulation import Simulation, Result
from .gsparser import build_network, load_file


from colomoto_jupyter import IN_IPYTHON
if IN_IPYTHON:
    from colomoto_jupyter import jupyter_setup
    from .widgets import *
    import ginsim


    jupyter_setup("mymodule",
        label="MaBoSS",
        color="blue", # for menu and toolbar
        menu=menu,
        toolbar=toolbar,
        js_api=js_api)