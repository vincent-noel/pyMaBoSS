pyMaBoSS
========

.. image:: https://maboss.curie.fr/images/maboss_logo.jpg
  :width: 300
  :alt: MaBoSS logo
  :align: center

pyMaBoSS is a python API for the `MaBoSS <https://maboss.curie.fr/>`_ software.
It can be used inside the `Colomoto docker <https://hub.docker.com/r/colomoto/colomoto-docker/>`_.
The API is loaded by running the following line in Python::
   
   import maboss
    
Once loaded, it allows you to quickly load a MaBoSS model::
   
   sim = maboss.load("metastasis.bnd", "metastasis.cfg")    
   
And then to simulate the model and plot the simulation results::
   
   res = sim.run()
   res.plot_piechart()

.. image:: https://raw.githubusercontent.com/colomoto/pyMaBoSS/master/doc/sample_piechart.png
  :width: 600
  :alt: Simulation result
  
  
Installation instructions
=========================
  
pyMaBoSS can be installed either via Conda, PyPi or GitHub. 


Installation with Conda
-----------------------

This is the recommended option, as it also allows the simple installation of all the dependencies. 

   conda -c colomoto install pymaboss
   
This will install the library, which will already be ready to use. 


Installation with PyPi
----------------------

This is not the recommended option as it cannot yet package the MaBoSS binaries, but if you already have then install you can just install pyMaBoSS using

   pip install maboss
   

To download the MaBoSS binaries, if you have conda and if you are using linux or macosx, you can run : 

   python -m maboss_setup
   
  
If you are using Windows, or if the command above did not work, you can try to run : 

   python -m maboss_setup_experimental
   
   
Otherwise, you can download them using the following links, for `Linux <https://github.com/sysbio-curie/MaBoSS-env-2.0/releases/latest/download/MaBoSS-linux64.zip>`_, `MacOSX arm64 <https://github.com/sysbio-curie/MaBoSS-env-2.0/releases/latest/download/MaBoSS-osx-arm64.zip>`_, `MacOSX X86_64 <https://github.com/sysbio-curie/MaBoSS-env-2.0/releases/latest/download/MaBoSS-osx64.zip>`_ or `Windows <https://github.com/sysbio-curie/MaBoSS-env-2.0/releases/latest/download/MaBoSS-win64.zip>`_. Once downloaded, you need to extract them and make them accessible by putting them in a folder configured in your PATH. 

Installation with GitHub
------------------------

Finally, you can also install pyMaBoSS directly from the official GitHub repository : 

   git clone https://github.com/colomoto/pyMaBoSS
   
   cd pyMaBoSS
   
   python setup.py install
   
Note that this also comes without the MaBoSS binaries, so to install them you'll have to follow the steps described above. 
