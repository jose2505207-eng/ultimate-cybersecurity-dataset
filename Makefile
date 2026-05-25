PYTHONPATH := src
export PYTHONPATH

.PHONY: test validate smoke inventory build-silver build-gold build-gold-benchmark evaluate-benchmark sample-10k sample-50k sample-100k clean

test:
	pytest

validate:
	python -m cyberdataset.validation --input data/gold_unified/ultimate_cybersecurity_dataset.csv

smoke:
	python -m cyberdataset.build.build_silver --smoke
	python -m cyberdataset.build.build_gold
	python -m cyberdataset.validation --input data/gold_unified/ultimate_cybersecurity_dataset.csv

inventory:
	python -m cyberdataset.inventory

build-silver:
	python -m cyberdataset.build.build_silver

build-gold:
	python -m cyberdataset.build.build_gold

build-gold-benchmark:
	python -m scripts.build_gold_benchmark --silver-dir data/silver_normalized --out-dir data/gold --max-rows 100000 --format both

evaluate-benchmark:
	@test -n "$(PREDICTIONS)" || (echo "Set PREDICTIONS=path/to/predictions.csv"; exit 1)
	python -m scripts.evaluate_benchmark --gold-file data/gold/benchmark_gold.csv --predictions-file "$(PREDICTIONS)" --out-dir data/gold

sample-10k:
	python -m cyberdataset.build.build_sample_10k

sample-50k:
	python -m cyberdataset.build.build_sample_50k

sample-100k:
	python -m cyberdataset.build.build_sample_100k

clean:
	rm -f data/silver_normalized/*.csv data/silver_normalized/*.parquet data/gold_unified/*.csv data/gold_unified/*.parquet data/reports/*.json
