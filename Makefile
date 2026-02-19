.PHONY: install sanity eval web cli check clean help

## Install all dependencies
install:
	pip install -r requirements.txt

## Run end-to-end sanity check → produces artifacts/sanity_output.json
sanity:
	python scripts/run_sanity.py

## Run automated evaluation harness → artifacts/eval_report.json
eval:
	python scripts/eval_harness.py --report

## Start the Streamlit web interface (streaming enabled)
web:
	streamlit run web_app.py

## Start the CLI streaming chat interface
cli:
	python cli.py chat

## Validate sanity_output.json format
check:
	python scripts/verify_output.py

## Show knowledge base stats
stats:
	python cli.py stats

## Remove ChromaDB and generated artifacts
clean:
	rm -rf chroma_db/
	rm -f artifacts/sanity_output.json artifacts/eval_report.json

## Show all commands
help:
	@echo ""
	@echo "  make install   Install dependencies"
	@echo "  make sanity    End-to-end test  → artifacts/sanity_output.json"
	@echo "  make eval      Evaluation harness → artifacts/eval_report.json"
	@echo "  make web       Streamlit web UI (streaming)"
	@echo "  make cli       CLI chat (streaming)"
	@echo "  make check     Validate sanity output JSON"
	@echo "  make stats     Knowledge base stats"
	@echo "  make clean     Remove generated files"
	@echo ""