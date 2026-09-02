.PHONY: install test eval benchmark run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

eval:
	python3 ops/eval_runner.py

benchmark:
	python3 ops/benchmark.py

run:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf __pycache__ .pytest_cache *.sqlite data/*.sqlite BENCHMARK_REPORT.md
