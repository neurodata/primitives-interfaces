CC=python3
all: py3

py3:
	$(CC) setup.py build

install:
	$(CC) setup.py install

dist:
	$(CC) setup.py sdist

up-test:
	twine upload dist/* -r testpypi
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

up:
	twine upload dist/*

clean:
	rm -rf build
	rm -rf MANIFEST
	rm -rf dist
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf *.egg-info/
	python3.6 setup.py clean --all
