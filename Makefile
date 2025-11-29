# -------------------------------
# Python Virtual Environment Makefile
# -------------------------------

# Name of the virtual environment directory
VENV_NAME = .venv

PREFIX = ~/Documents/code

# Python interpreter to use
PYTHON = python3.12

# Requirements file
REQ = requirements.txt

# Directory where the venv bin directory lives
BIN = $(PREFIX)/$(VENV_NAME)/bin

# -------------------------------
# Create venv
# -------------------------------
venv:
	@if [ ! -d "$(PREFIX)/$(VENV_NAME)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(PREFIX)/$(VENV_NAME); \
		echo "Created venv in $(PREFIX)/$(VENV_NAME)"; \
	else \
		echo "Virtual environment already exists."; \
	fi

# -------------------------------
# Install dependencies
# -------------------------------
install: venv
	@if [ -f "$(REQ)" ]; then \
		echo "Installing requirements..."; \
		$(BIN)/pip install --upgrade pip; \
		$(BIN)/pip install -r $(REQ); \
	else \
		echo "No requirements.txt found; skipping installation."; \
	fi

# -------------------------------
# Activate venv (prints instructions)
# -------------------------------
activate:
	@echo "To activate the virtual environment, run:"
	@echo "    source $(BIN)/activate"

# -------------------------------
# Remove venv
# -------------------------------
clean:
	@if [ -d "$(PREFIX)/$(VENV_NAME)" ]; then \
		echo "Removing virtual environment..."; \
		rm -rf $(PREFIX)/$(VENV_NAME); \
		echo "Removed."; \
	else \
		echo "No venv to remove."; \
	fi

# -------------------------------
# Full reset: clean + create + install
# -------------------------------
reset: clean venv install
	@echo "Environment reset complete."

# -------------------------------
# Freeze dependencies to requirements.txt
# -------------------------------
freeze:
	$(BIN)/pip freeze > $(REQ)
	@echo "Dependencies written to $(REQ)."

.PHONY: venv install clean reset activate freeze
