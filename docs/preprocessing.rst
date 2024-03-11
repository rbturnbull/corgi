=============
Preprocessing
=============

Use this tutorial to learn how to generate the files needed to train Corgi from scratch. 
You can use this to replicate the results found in the paper (still to be released).

Corgi takes two input files, a SeqTree and a SeqBank. 
The SeqBank holds the sequence data for each accession in the dataset. 
The SeqTree has the information about the cross-validation partition for each accession and which node in the taxonomic tree that the accession corresponds to.

To create the initial SeqTree object, run this command:

.. code-block:: bash

    refseq-to-seqtree corgi-max200-initial.st --max-seqs 200 --accessions corgi-max200-initial.txt --partitions 6

This will download the latest version of the RefSeq catalog and create a SeqTree file ``corgi-max200.st`` which include at most 200 accessions for each order. 

Corgi was trained with the 221 release of RefSeq. If you want to explicitly give it a RefSeq catalog, then you can specify the path to it with the ``--catalog`` option. 

The list of accessions was saved to the file ``corgi-max200-initial.txt``. This list can be used to create a SeqBank file:

Each genome will be assigned to one of six partitions. We will use partition 0 as the test set and partitions 1–5 as the cross-validation partitions.

.. code-block:: bash

    seqbank refseq corgi-max200.sb --filter corgi-max200-initial.txt

This command will download each raw RefSeq file and save the data for each accession in ``corgi-max200-initial.txt``.

There may be accessions found in the catalog that were not found in the seqbank file. A new SeqTree file can be created with these removed:

.. code-block:: bash

    seqtree intersection-seqbank corgi-max200-initial.st corgi-max200.sb corgi-max200.st

