# Contribution Guide

We welcome and encourage community contributions to lingam package.

There are many ways to help lingam:

* Implement a feature
* Send a patch
* Report a bug
* Fix/Improve documentation
* Write examples and tutorials

## Code Style

We try to closely follow the official Python guidelines detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/). Please read it and follow it.

In addition, we add the following guidelines:

* Use underscores to separate words in non class names: n_samples rather than nsamples.
* Use relative imports for references inside lingam package.
* Use the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide) in all your docstrings.

## Checking the Format

Coding style is checked with flake8.

``` sh
flake8 lingam --count --ignore=E203,E741 --max-complexity=10 --max-line-length=127 --statistics
```

## Documentation

When adding a new feature to lingam, you also need to document it in the reference. The documentation source is stored under the docs directory and written in reStructuredText format. The API reference is automatically output from the docstring.

To build the documentation, you use Sphinx. Run the following commands to install Sphinx and its extensions.

``` sh
pip install sphinx
pip install sphinxcontrib-napoleon
pip install sphinx_rtd_theme
```

Then you can build the documentation in HTML format locally:

``` sh
cd docs
make html
```

HTML files are generated under build/html directory. Open index.html with the browser and see if it is rendered as expected.

## Unit Tests

When adding a new feature or fixing a bug, you also need to write sufficient test code. We use pytest as the testing framework and unit tests are stored under the tests directory. We check that the code coverage is 100% when we run pytest.

You can run all your tests as follows:

``` sh
pytest -v --cov=lingam --cov-report=term-missing tests
```

## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind:

### Title

The title of your pull request should

* briefly describe and reflect the changes
* wrap any code with backticks
* not end with a period

### Description

The description of your pull request should

* describe the motivation
* describe the changes
* if still work-in-progress, describe remaining tasks
