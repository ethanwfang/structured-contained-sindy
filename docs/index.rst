Structure-Constrained SINDy
============================

**Structure-Constrained SINDy** (SC-SINDy) is a method for discovering governing
equations from data using learned structural priors.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/getting_started
   tutorials/custom_systems
   tutorials/training_networks
   tutorials/real_world_data
   theory/sindy_overview
   theory/structure_constraints
   api/index

Quick Start
-----------

Installation::

    pip install sc-sindy

Basic usage:

.. code-block:: python

    from sc_sindy import sindy_stls, build_library_2d, VanDerPol

    # Generate data
    system = VanDerPol(mu=1.0)
    t, X = system.simulate([1.0, 0.0], t_span=(0, 10), dt=0.01)

    # Build library and discover equations
    Theta, labels = build_library_2d(X)
    xi, _ = sindy_stls(Theta, X_dot, threshold=0.1)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
