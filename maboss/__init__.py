from .network import *
from .simulation import *
from .result import *
from .gsparser import load
from .sbml import *

from colomoto_jupyter import IN_IPYTHON
if IN_IPYTHON:
    from colomoto_jupyter import jupyter_setup
    from .widgets import *

    import matplotlib.pyplot as plt
    plt.ion()

    jupyter_setup("mymodule",
        label="MaBoSS",
        color="green", # for menu and toolbar
        menu=menu,
        toolbar=toolbar,
        js_api=js_api)

