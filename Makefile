PYTHON_VERSION="3.11"

mypy:
	mypy src/modelling --config-file mypy.ini --python-version=${PYTHON_VERSION}