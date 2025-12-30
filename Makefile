PYTHON_VERSION="3.11"

mypy:
	mypy src/modelling --config-file mypy.ini --python-version=${PYTHON_VERSION}

down:
	docker compose down

up: down 
	docker compose up --build -d