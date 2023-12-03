CODE = skuld
TESTS = tests

format:
	ruff format $(CODE) $(TESTS)
	ruff --fix $(CODE) $(TESTS)

lint:
	ruff format --check $(CODE)
	ruff check $(CODE)

test:
	pytest $(TESTS)
