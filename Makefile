all: py3

py3:
	python3 setup.py build

install:
	python setup.py install

dist:
	python setup.py sdist

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
