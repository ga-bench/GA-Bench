# GA-Bench

Code for **GA-Bench: A Benchmark for Completeness and Readability of Scientific
Graphical Abstracts** (JCDL '26).

This repository contains the extraction, scoring, and analysis pipeline. The
dataset (10,000 paper–GA pairs with metadata, extracted content, and model
outputs) is hosted separately on Hugging Face.

## Structure

- `extraction/` — PDF-to-structured-content pipeline (GROBID + pdffigures2:
  full text, IMRaD sections, figures, tables, equations, quality reports).
- `dataset_build/` — dataset statistics, selection, and packaging scripts.
- `prompts/` — SRP and grounding prompt builders.
- `task1_completeness/` — Structured Reference Profiles (variants A/B),
  GA grounding, deterministic S/R/C/L scoring, and the naive baseline.
- `task2_readability/` — OCR text-legibility, visual-clarity, and VLM
  semantic-interpretability feature extraction plus readability scoring.

## Models

Three open vision–language models: Qwen3-VL-32B-Instruct, Gemma-3-27B-IT, and
Mistral-Small-3.1-24B-Instruct (served locally via vLLM).

## Notes

- Paths in scripts are relative placeholders (`./`); set them to your own
  data locations before running.
- Jobs are provided as PBS scripts for an HPC scheduler; adapt to your cluster.
- Inference uses a local vLLM OpenAI-compatible endpoint (`api_key="not-needed"`).

## Data

The dataset is available on Hugging Face (link omitted for anonymous review).
