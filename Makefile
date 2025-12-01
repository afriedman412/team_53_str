# ---------------------------
# Settings
# ---------------------------

# Where you want the venv to live (outside the repo)

VENV := venv

PYTHON := $(VENV)/bin/python3.12
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

APP := app.main:app
REQUIREMENTS := requirements.txt

# ---------------------------
# PHONY targets
# ---------------------------
.PHONY: venv install run test clean reset

# ---------------------------
# Create venv (in ~/Documents/code)
# ---------------------------
venv:
	@echo "🔧 Creating Python 3.12 virtual environment at $(VENV)..."
	python3.12 -m venv $(VENV)
	@echo "📦 Upgrading pip..."
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

# ---------------------------
# Install dependencies
# ---------------------------
install: venv
	@echo "📦 Installing dependencies..."
	$(PIP) install -r $(REQUIREMENTS)

# ---------------------------
# Run the API
# ---------------------------
run:
	@echo "🚀 Starting FastAPI server..."
	$(UVICORN) $(APP) --reload --port 8000

# ---------------------------
# Run tests with Python 3.12
# ---------------------------
test:
	@echo "🧪 Running tests with Python 3.12..."
	$(PYTHON) -m pytest -vv --disable-warnings

# ---------------------------
# Clean venv in ~/Documents/code
# ---------------------------
clean:
	@echo "🧹 Removing external venv at $(VENV)..."
	rm -rf $(VENV)

# ---------------------------
# Full reset (clean + reinstall)
# ---------------------------
reset: clean install
	@echo "✨ Environment reset complete!"
