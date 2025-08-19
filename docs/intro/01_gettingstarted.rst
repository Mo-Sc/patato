Installation
================
In order to use PATATO, you must have a Python environment set up on your computer. We recommend using uv (https://docs.astral.sh/uv/) to run Python. This will help you to avoid dependency conflicts. You can alternatively use Anaconda or virtual environments. 

You can install patato with uv like so:

.. code-block:: bash

    uv add patato


Or, using pip:

.. code-block:: bash

    pip install patato

To setup support for GPU-based reconstruction, please follow the installation guide in the documentation.
.. tip::
    If you are using Anaconda, you may wish to create a new environment before installing PATATO. This can be
    done by running the following command in the Anaconda prompt:

        conda create -n patato python=3.11

    Then activate the environment by running:

        conda activate patato

    You can then install PATATO as normal.

Installation
+++++++++++++

Option 1: Install using uv
------------------------------------------------------

Once you have Python installed and uv setup, you can install PATATO using uv:

.. code-block:: bash
   :caption: Install PATATO using uv.

        uv add patato

To add GPU support, follow the guide on the JAX official guide here: (https://github.com/google/jax#installation).

Option 2: Install using pip
------------------------------------------------------

Once you have Python installed, you can install PATATO using pip:

.. code-block:: bash
   :caption: Install PATATO using pip.

        pip install patato

To add GPU support, follow the guide on the JAX official guide here: (https://github.com/google/jax#installation).

Option 3: Install from source
------------------------------------

To install the most recent version of PATATO from GitHub:

.. code-block:: bash
   :caption: Install PATATO from GitHub.

    pip install git+https://github.com/BohndiekLab/patato

Option 4: Install from source (editable)
----------------------------------------------------------

To install the development version of PATATO from GitHub and allow editing for development purposes:

.. code-block:: bash
   :caption: Install PATATO from source.

        cd /path/to/installation/directory
        git clone https://github.com/BohndiekLab/patato
        cd patato
        pip install -e .
