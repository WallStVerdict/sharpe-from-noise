# sharpe-from-noise: `make reproduce` builds a local venv on first use, then runs every number and chart.
VENV   ?= .venv
DOCKER ?= docker

ifdef IN_DOCKER
PY    := python
STAMP :=
else
PY    := $(VENV)/bin/python
STAMP := $(VENV)/.installed
endif

.PHONY: reproduce docker clean

# Part 1: the claim. Part 2: hold-out + walk-forward. Extra: the real US market (downloads the Ken French file).
reproduce: $(STAMP)
	@$(PY) noise_sharpe.py
	@$(PY) walk_forward.py
	@$(PY) real_market.py

$(VENV)/.installed: requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --disable-pip-version-check -r requirements.txt
	@touch $@

# Run `make reproduce` in a clean container: no local Python needed. DOCKER=podman works too.
docker:
	$(DOCKER) build -t sharpe-from-noise .
	$(DOCKER) run --rm -v "$$PWD/results:/app/results" sharpe-from-noise

clean:
	rm -rf results
