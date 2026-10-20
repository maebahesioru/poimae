# PoiMae - one command per job, because a judge should not have to read docs
# to run the thing.
#
#   make            # what it does when you type nothing
#   make run        # local web app on http://127.0.0.1:8765
#   make test       # pytest
#   make eval       # regenerate the synthetic corpus and re-measure
#   make video      # rebuild the demo video from slides + screenshots
#   make clean      # throw away generated artefacts (never data/)

SHELL := /bin/bash
PY := .venv/bin/python
PORT ?= 8765
export OMP_NUM_THREADS := 2

.PHONY: help setup run test eval corpus shots video clean lint

help:
	@sed -n '1,12p' Makefile | sed 's/^# \{0,1\}//'

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

run:
	@echo "PoiMae: http://127.0.0.1:$(PORT)  (Ctrl-C to stop)"
	$(PY) -m uvicorn poimae.app:app --host 127.0.0.1 --port $(PORT) --app-dir src

test:
	$(PY) -m pytest

eval: corpus
	$(PY) -m tools.evaluate --corpus eval/synthetic --split test \
		--json eval/report-test.json --markdown eval/CONFUSION.md

corpus:
	$(PY) -m tools.synth --out eval/synthetic --per-class 12

shots:
	$(PY) -m tools.screenshots --out evidence

video:
	./video/make_video.sh

lint:
	$(PY) -m compileall -q src tools tests

clean:
	rm -rf eval/synthetic .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
