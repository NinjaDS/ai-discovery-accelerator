# 🚀 AI Discovery Accelerator

> **Multi-agent AI system for rapid ML opportunity discovery** — powered by Amazon Bedrock (Claude) and a sequential pipeline of intelligent agents for data profiling, gap analysis, feature intelligence, and model selection.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

---

## Overview

The AI Discovery Accelerator automates the most time-consuming parts of every ML project kickoff: understanding your data, identifying gaps, ranking features, selecting the right model, and generating a boardroom-ready discovery report — all in a single CLI command.

```
python main.py --sample churn_prediction
```

In under 60 seconds you get a full ML Discovery Report: data quality scores, feature rankings, model recommendations, and LLM-generated natural language insights from Amazon Bedrock Claude.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       AI Discovery Accelerator                          │
│                                                                         │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────────┐            │
│  │ DataLoader │───▶│ DataProfiler │───▶│   GapAnalyst     │            │
│  │            │    │              │    │                  │            │
│  │ CSV / JSON │    │ null rates   │    │ readiness score  │            │
│  │ Excel /    │    │ type infer   │    │ missing fields   │            │
│  │ Synthetic  │    │ outliers     │    │ recommendations  │            │
│  └────────────┘    └──────────────┘    └────────┬─────────┘            │
│                                                 │                       │
│  ┌──────────────────────┐    ┌──────────────────▼──────────────────┐   │
│  │  InsightSynthesiser  │◀───│  ModelSelector  FeatureIntelligence │   │
│  │                      │    │                                     │   │
│  │ Amazon Bedrock       │    │ top 3 models    RF importance       │   │
│  │ Claude 3.5 Sonnet    │    │ training strat  mutual info         │   │
│  │ (+ template fallback)│    │ eval metrics    transform recs      │   │
│  └──────────┬───────────┘    └─────────────────────────────────────┘   │
│             │                                                           │
│             ▼                                                           │
│  ┌──────────────────────┐                                              │
│  │       Report         │  JSON + Markdown saved to output/            │
│  │  executive summary   │  Rich-formatted console output               │
│  │  key findings        │                                              │
│  │  next steps          │                                              │
│  └──────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/kumaresaperumal/ai-discovery-accelerator.git
cd ai-discovery-accelerator
pip install -r requirements.txt
```

### 2. Configure AWS (for Bedrock)

```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

```ini
# .env
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
```

> **Note:** If Bedrock is unavailable, the system gracefully falls back to a template-based report. All agents still run fully.

### 3. Run with Synthetic Data

```bash
# Churn prediction
python main.py --sample churn_prediction

# Fraud detection
python main.py --sample fraud_detection

# Sales forecasting
python main.py --sample sales_forecasting --task-type regression
```

### 4. Run with Your Own Data

```bash
python main.py \
  --dataset data/customers.csv \
  --use-case churn_prediction \
  --task-type classification \
  --output ./output
```

---

## Usage Examples

```bash
# Analyse a CSV file for churn
python main.py --dataset customers.csv --use-case churn_prediction

# Fraud detection on a JSON dataset
python main.py --dataset transactions.json --use-case fraud_detection --task-type classification

# Sales forecasting from Excel
python main.py --dataset sales.xlsx --use-case sales_forecasting --task-type regression

# Quiet mode (outputs file paths only, for scripting)
python main.py --sample churn_prediction --quiet

# Specify a different AWS region
python main.py --sample fraud_detection --aws-region eu-west-1

# Multi-source enterprise data
python main.py --dataset crm_billing.csv --use-case churn_prediction --multi-source
```

---

## Agents

| Agent | Role | Output |
|-------|------|--------|
| **DataLoader** | Load CSV/JSON/Excel/Parquet or generate synthetic data | `pd.DataFrame` |
| **DataProfiler** | Null rates, type inference, outlier detection, distribution stats | `data_profile` dict |
| **GapAnalyst** | Score data readiness 0-100 against use-case feature requirements | `gap_report` dict |
| **FeatureIntelligence** | RF feature importance, mutual information, correlation, transform recs | `feature_intelligence` dict |
| **ModelSelector** | Recommend top-3 ML models with rationale, effort estimates, eval strategy | `model_recommendations` dict |
| **InsightSynthesiser** | LLM-powered natural language report via Bedrock Claude (+ template fallback) | `insights` dict |

---

## Supported Use Cases

| Use Case | Task Type | Sample Rows | Target Column |
|----------|-----------|-------------|---------------|
| `churn_prediction` | classification | 1,000 | `churn` |
| `fraud_detection` | classification | 1,000 | `is_fraud` |
| `sales_forecasting` | regression | 500 | `revenue` |

---

## Output

Reports are saved to `./output/` (configurable with `--output`):

```
output/
├── report_20250101_120000.json   # Full structured report (all agent outputs)
└── report_20250101_120000.md    # Markdown discovery report
```

The JSON report contains:
- `metadata` — run timestamp, version, use case
- `data_profile` — full column-level statistics
- `gap_report` — readiness score, missing fields, risks
- `feature_intelligence` — importance rankings, correlations
- `model_recommendations` — top-3 models with rationale
- `insights` — executive summary, key findings, next steps

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM / NLG | Amazon Bedrock — Claude 3.5 Sonnet |
| Data Processing | pandas, numpy |
| ML / Statistics | scikit-learn, scipy |
| CLI & Display | Rich |
| Synthetic Data | Faker, numpy.random |
| AWS SDK | boto3 |
| Testing | pytest |
| Packaging | pyproject.toml / setuptools |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
ai-discovery-accelerator/
├── core/
│   ├── __init__.py
│   ├── utils.py                    # Logging, file I/O, Rich helpers
│   ├── data_loader.py              # Data loading + synthetic generation
│   ├── orchestrator.py             # Sequential pipeline orchestrator
│   └── agents/
│       ├── __init__.py
│       ├── data_profiler.py        # Data Profiling Agent
│       ├── gap_analyst.py          # Gap Analysis Agent
│       ├── feature_intelligence.py # Feature Intelligence Agent
│       ├── model_selector.py       # Model Selection Agent
│       └── insight_synthesiser.py  # LLM Synthesis Agent (Bedrock)
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_data_profiler.py
│   └── test_feature_intelligence.py
├── main.py                         # CLI entry point
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## License

MIT License — Copyright (c) 2025 Kumaresa Perumal. See [LICENSE](LICENSE) for details.
