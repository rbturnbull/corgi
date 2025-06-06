.. image:: https://raw.githubusercontent.com/rbturnbull/corgi/main/docs/images/corgi-banner.jpg

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge| |git3moji badge| |torchapp badge|

.. |testing badge| image:: https://github.com/rbturnbull/corgi/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/corgi/actions

.. |docs badge| image:: https://github.com/rbturnbull/corgi/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/corgi
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/ee1b52dd314d6441e0aabc0e1e50dc2c/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/corgi/coverage/

.. |git3moji badge| image:: https://img.shields.io/badge/git3moji-%E2%9A%A1%EF%B8%8F%F0%9F%90%9B%F0%9F%93%BA%F0%9F%91%AE%F0%9F%94%A4-fffad8.svg
    :target: https://robinpokorny.github.io/git3moji/

.. |torchapp badge| image:: https://img.shields.io/badge/torch-app-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
        
.. end-badges

.. start-quickstart

Installation
============

The software can be installed using ``pip``

.. code-block:: bash

    pip install bio-corgi

.. warning ::

    Do not try just ``pip install corgi`` because that is a different package.

To install the latest version from the repository, you can use this command:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/corgi.git

.. note ::

    Soon corgi will be able to be installed using conda.

Usage
============

To make predictions, run the ``corgi`` command line tool:

.. code-block:: bash

    corgi --input <input seq file> --output-csv <output csv file>

This will output all the predictions in a CSV file.

For more information about the other options, see the help with:

.. code-block:: bash

    corgi --help

For help on training a model with corgi, run:

.. code-block:: bash

    corgi-tools --help

.. end-quickstart


Credits
==================================

* `Robert Turnbull <https://robturnbull.com>`_
* Vinícius Salazar
* Edoardo Tescari
* Heroen Verbruggen

With contributions from:

* Yuhao Tong
* Kamalpreet Singh
* Rafsan Al Mamun
* Diego Aranda Villarreal

Citation details to come.

* Created using torchapp (https://github.com/rbturnbull/torchapp)

