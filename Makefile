PY ?= .venv/bin/python
SCENE ?= a

.PHONY: setup demo bench run clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# synthetic room with known changes: render (once), detect, score
demo:
	@test -f data/synthetic/gt.json || $(PY) tools/make_synthetic.py --scene a --out data/synthetic
	$(PY) -m rayvote data/synthetic/before.mp4 data/synthetic/after.mp4 --out out/demo
	$(PY) tools/eval_synthetic.py --gt data/synthetic/gt.json --run out/demo
	@echo "open out/demo/viewer.html and out/demo/report.md"

# both synthetic scenes
bench: demo
	@test -f data/synthetic_b/gt.json || $(PY) tools/make_synthetic.py --scene b --out data/synthetic_b
	$(PY) -m rayvote data/synthetic_b/before.mp4 data/synthetic_b/after.mp4 --out out/demo_b
	$(PY) tools/eval_synthetic.py --gt data/synthetic_b/gt.json --run out/demo_b

# your own recordings
run:
	$(PY) -m rayvote data/before.mp4 data/after.mp4 --out out/room

clean:
	rm -rf out
