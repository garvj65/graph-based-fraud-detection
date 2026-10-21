.PHONY: stage1 test

stage1:
	python -m pytest -q
	python -m src.data --config configs/default.yaml --download

test:
	python -m pytest -q
