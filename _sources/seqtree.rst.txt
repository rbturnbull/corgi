==========
SeqTree
==========


One of the input files that Corgi uses for training is a SeqTree.
This works like a dictionary in Python where the keys are the accessions in the dataset and the value is a SeqDetail which has the index of the cross-validation partition to use for the accession as well as the taxonomic tree that the accession corresponds to.

SeqTree objects can be worked on using ``seqbank`` on the command line.

To see the options available, run:

.. code-block:: bash

    seqbank --help


.. click:: corgi.seqtree:app
   :prog: seqtree
   :nested: full

