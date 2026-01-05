PYTHON_VERSION="3.11"

mypy:
	mypy modules --config-file mypy.ini --python-version=${PYTHON_VERSION}

build:
	docker compose build

down:
	docker compose down

train: down 
	docker compose run --rm \
		-e BATCH_SIZE=4 \
		-e EPOCHS=2 \
		-e ON_GPU=false \
		model-training