.. _model_reference:
======
Models
======

Franken Machine Learning Potential
----------------------------------

.. autoclass:: franken.rf.model.FrankenPotential
   :members: energy, energy_and_forces, feature_map, grad_energy_func, grad_energy_autograd
   :special-members:

.. _random_features_params:
Random Features heads
---------------------
.. autoclass:: franken.rf.heads.RandomFeaturesHead
   :special-members:

.. autoclass:: franken.rf.heads.Linear
   :special-members:

.. autoclass:: franken.rf.heads.OrthogonalRFF
   :special-members:

.. autoclass:: franken.rf.heads.TensorSketch
   :special-members: