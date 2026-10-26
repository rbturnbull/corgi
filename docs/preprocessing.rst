=============
Preprocessing
=============

Use this tutorial to learn how to generate the files needed to train Corgi from scratch. 
You can use this to replicate the results found in the paper (still to be released).

Corgi takes two input files – a SeqTree and a SeqBank. 
The SeqBank holds the sequence data for each accession in the dataset. 
The SeqTree has the information about the cross-validation partition for each accession and which node in the taxonomic tree that the accession corresponds to.

Create the SeqBank
==================

First, create a SeqBank with the sequences from RefSeq using the `seqbank <https://github.com/rbturnbull/seqbank>`_ command-line tool:

.. code-block:: bash

    seqbank refseq refseq.sb

This will download each raw RefSeq file and save the data for each accession.
By default, the SeqBank is stored with the RocksDict engine. Use the ``--engine`` option to choose another (see :ref:`seqbank-engines` below).

Create the SeqTree
==================

Next, create the SeqTree and a SeqBank restricted to the accessions it contains:

.. code-block:: bash

    corgi-tools refseq-to-seqtree \
        --output corgi-max200.st \
        --seqbank refseq.sb \
        --seqbank-output corgi-max200.sb \
        --max-seqs 200 \
        --partitions 6 \
        --accessions corgi-max200.txt

This will download the latest version of the RefSeq catalog and create a SeqTree file ``corgi-max200.st`` which includes at most 200 accessions for each order. 
Only accessions found in ``refseq.sb`` are included.
The sequences for these accessions are copied to the new SeqBank ``corgi-max200.sb`` and the list of accessions is saved to ``corgi-max200.txt``.

Each accession will be assigned to one of six partitions. We will use partition 0 as the test set and partitions 1–5 as the cross-validation partitions.

Corgi was trained with the 225 release of RefSeq. If you want to explicitly give it a RefSeq catalog, then you can specify the path to it with the ``--catalog`` option. 

.. _seqbank-engines:

Choosing the SeqBank storage engine
-----------------------------------

By default, the output SeqBank is stored with the `RocksDict <https://github.com/rocksdict/RocksDict>`_ engine. 
Use the ``--engine`` option to choose a different one:

.. code-block:: bash

    corgi-tools refseq-to-seqtree \
        --output corgi-max200.st \
        --seqbank refseq.sb \
        --seqbank-output corgi-max200.sb \
        --engine lmdb

The available engines are:

``rocksdict``
    The default.

``lmdb``
    `LMDB <https://lmdb.readthedocs.io/>`_, a memory-mapped key-value store.

``speedict``
    The legacy engine used by older SeqBanks. 
    It requires installing ``seqbank[speedict]`` and is only available for Python versions below 3.13.

The ``--engine`` option only applies when a new SeqBank is created. 
When Corgi opens an existing SeqBank, it detects the engine automatically, 
so no option is needed for training or for the other commands. 
SeqBanks created before engines were introduced are treated as ``speedict`` banks.

To change the engine of an existing SeqBank, use the ``seqbank convert`` command, which copies it to a new directory:

.. code-block:: bash

    seqbank convert corgi-max200.sb corgi-max200-lmdb.sb --engine lmdb
