# Setup dependencies (installing required packages)
setup:
	pip install -r requirements.txt

# Format the code with black
format:
	black .
	isort .

# Lint the code with flake8
lint:
	flake8 .

# Test step (currently no tests, placeholder)
test:
	echo "No tests available. Add tests to the 'tests/' folder."
