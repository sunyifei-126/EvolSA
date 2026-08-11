.PHONY: test audit demo

test:
	python -m unittest discover -s tests -v

audit:
	python scripts/check_release.py

demo:
	evolsa demo --output .evolsa/demo
