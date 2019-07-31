.. api documentation.

maboss API
==========

In the ``maboss`` module, the informations contained in the *MaBoSS*'s ``.cfg`` and ``.bnd`` files are represented by Python objects.

:doc:`network` objects represent most of the information contained in the ``.bnd``
file, and :doc:`simulation` object contained the information contained in the ``.cfg`` file.

The ``maboss`` module can call the *MaBoSS* software via the ``run`` method of
``Simulation`` objects. This method writes a ``.cfg`` and a ``.bnd`` files and
run *MaBoSS* on them. Then, a :doc:`result` object is created to interact with *MaBoSS* output files.

.. toctree::
   :maxdepth: 1

   load.rst
   network.rst
   simulation.rst
   result.rst