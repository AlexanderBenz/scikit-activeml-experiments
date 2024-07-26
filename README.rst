# scikit-activeml-experiments
`github pages <https://alexanderbenz.github.io/scikit-activeml-experiments/>`_

Open app localy
``shiny run src_shiny_app/app.py``


Open server plots localy
``shiny static-assets remove``
``shinylive export src_shiny_app docs``
``python -m http.server --directory docs --bind localhost 8008``

Installation
Please install `pytorch <https://pytorch.org/get-started/locally/>`_ using the installation guide.
