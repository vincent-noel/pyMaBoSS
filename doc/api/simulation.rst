.. simulation:

Simulation
==========

The ``maboss`` module can call the *MaBoSS* software via the ``run`` method of
``Simulation`` objects. This method writes a ``.cfg`` and a ``.bnd`` files and
run *MaBoSS* on them. Then, a ``Result`` object is created to interact with *MaBoSS* output files.

* **Simulation**

.. autoclass:: maboss.simulation.Simulation
   :members:
   :special-members: __init__

