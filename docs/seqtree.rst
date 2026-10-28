==========
SeqTree
==========


One of the input files that Corgi uses for training is a SeqTree.
This works like a dictionary in Python where the keys are the accessions in the dataset and the value is a SeqDetail which has the index of the cross-validation partition to use for the accession as well as the taxonomic tree that the accession corresponds to.

SeqTree objects can be worked on using ``seqtree`` on the command line.

To see the options available, run:

.. code-block:: bash

    seqtree --help

For example, to print the tree with the number of accessions at each node and the children sorted alphabetically:

.. code-block:: bash

    seqtree render corgi.st --count --alphabetical

To plot the tree as a sunburst chart, use ``seqtree sunburst``. 
Saving to an ``.html`` file works with a standard installation. 
Saving a static image (e.g. ``.png``, ``.svg`` or ``.pdf``) requires the ``images`` extra (``pip install "bio-corgi[images]"``) 
and a Chrome installation for Kaleido:

.. code-block:: bash

    seqtree sunburst corgi.st --output corgi.html

Use the ``seqtree export`` command to write the sequences of a SeqTree from a SeqBank to a file with the classification of each sequence in the sequence ID.


.. click:: corgi.seqtree:app
   :prog: seqtree
   :nested: full

