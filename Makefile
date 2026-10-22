.PHONY: stage1 stage2 stage3 stage4 stage5 stage6 test

stage1:
	python -m pytest -q
	python -m src.data --config configs/default.yaml --download

test:
	python -m pytest -q

stage2:
	python -m pytest -q
	python -m src.train --config configs/stage2.yaml --download

stage3:
	python -m pytest -q
	python -m src.graph_train --config configs/stage3.yaml --download

stage4:
	python -c "import torch, torch_geometric"
	python -m pytest -q
	python -m src.gnn_train --config configs/stage4.yaml --download
	python scripts/validate_stage4.py
	python scripts/summarize_stage4.py

stage5:
	python -c "import torch, torch_geometric"
	python -m pytest -q
	python -m src.final_evaluate --config configs/stage5.yaml
	python scripts/validate_stage5.py
	python scripts/summarize_stage5.py

stage6:
	python -m pytest -q
	python -m src.monitor --config configs/stage6.yaml
	python scripts/validate_stage6.py
	python scripts/summarize_stage6.py
