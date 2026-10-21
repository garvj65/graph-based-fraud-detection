.PHONY: stage1 stage2 test

stage1:
	python -m pytest -q
	python -m src.data --config configs/default.yaml --download

test:
	python -m pytest -q

stage2:
	python -m pytest -q
	python -m src.train --config configs/stage2.yaml --download
