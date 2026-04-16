# Personal Glucose Health Insight Agent

A glucose-focused extension of the Personal Health Insights Agent (PHIA), combining agentic reasoning with lightweight CGM-oriented processing to generate personalized, non-diagnostic glucose insights.

## Overview

This project is motivated by two complementary lines of work.

The first motivation comes from **Transforming wearable data into personal health insights using large language model agents**, which introduces PHIA as an agentic framework for personal health question answering. PHIA uses an iterative **Thought → Act → Observe** pipeline, together with tool use such as code generation and web search, to reason over personal health data. This repository builds directly on that agent design and adapts the PHIA-style pipeline to the glucose domain.

The second motivation comes from **Glucotypes reveal new patterns of glucose dysregulation**, which emphasizes that blood glucose should not be understood only through single measurements or coarse averages. Instead, continuous glucose monitoring (CGM) reveals temporal variability, postprandial excursions, and individualized response patterns that are important for understanding glucose dysregulation. This repository is informed by that glucose-specific perspective and treats CGM as time-series data rather than purely static tabular data.

Importantly, this project does **not** attempt to fully reproduce the glucotype paper’s original clustering-based methodology. Instead, it implements a **lightweight glucose processing pipeline** designed for downstream agent reasoning.

## Project Goal

The goal of this repository is to extend a general personal-health agent framework into a glucose-specific setting.

Instead of focusing on general wearable signals such as sleep and activity, this project centers on:

- continuous glucose monitoring (CGM)
- meal-linked glucose responses
- daily glucose summaries
- personalized glucose interpretation

The system is designed to support:

- postprandial spike analysis
- glucose pattern summarization
- risk-oriented but non-diagnostic interpretation
- personalized and actionable insight generation grounded in the user’s own data

## Core Idea

This project combines:

1. a **PHIA-inspired agent pipeline** for iterative reasoning over health-related data
2. a **glucose-oriented processing layer** for extracting interpretable structure from CGM data

The result is a glucose insight agent that can reason over meal-level and day-level glucose behavior through a ReAct-style workflow.

## Method Summary

The repository consists of three main layers.

### 1. Glucose data preprocessing

Raw CGM, meal, and profile data are transformed into structured intermediate tables for downstream reasoning.

### 2. Lightweight glucose feature and rule layer

Rather than reproducing the full glucotype clustering framework, this project uses a lightweight processing strategy centered on:

- **event-centered meal windows**
- **daily aggregation**
- **rule-based interpretation**

In particular, glucose responses are analyzed through meal-centered continuous windows around each meal event, together with day-level summaries. This means the project keeps a time-series perspective, but in a simpler and more interpretable form than a full overlapping sliding-window clustering pipeline.

### 3. Agent-based reasoning

On top of the structured glucose tables, a PHIA-style agent answers natural-language glucose questions using an iterative:

- **Thought**
- **Act**
- **Observe**

workflow.

The agent can use code execution to inspect glucose-derived tables, reason step by step over evidence, and generate personalized but non-diagnostic responses.

## What This Project Is

This repository **is**:

- a glucose-domain extension of the PHIA-style agent framework
- a lightweight CGM reasoning system
- a demonstration of how agentic personal-health pipelines can be adapted to blood glucose data


## Design Principles

This project is built around the following principles:

- **Agent-first reasoning**: preserve the PHIA-style Thought–Act–Observe pipeline
- **Time-series awareness**: treat CGM as temporal data rather than only static summary values
- **Meal-centered interpretation**: prioritize postprandial spike and recovery analysis
- **Higher-level table priority**: use structured features and summaries before falling back to raw traces
- **Personalization through evidence**: ground responses in the user’s own glucose data
- **Non-diagnostic scope**: generate insights and suggestions without claiming diagnosis

## Features

- CGM-centered preprocessing pipeline
- meal-level glucose feature extraction
- daily glucose summary generation
- lightweight glucose rule engine
- PHIA-style ReAct agent for glucose question answering
- personalized, explainable, and non-diagnostic insight generation
- optional background search for contextual health information

## References

1. Merrill, M. A., Paruchuri, A., Rezaei, N., Kovacs, G., Perez, J., Liu, Y., Schenck, E., Hammerquist, N., Sunshine, J., Tailor, S., et al. *Transforming wearable data into personal health insights using large language model agents*. *Nature Communications* (2026).  [oai_citation:2‡s41467-025-67922-y.pdf](sediment://file_0000000050e871fd83b8fd30763379c6)  
2. Hall, H., Perelman, D., Breschi, A., Limcaoco, P., Kellogg, R., McLaughlin, T., Snyder, M., et al. *Glucotypes reveal new patterns of glucose dysregulation*. *PLOS Biology* 16(7): e2005143 (2018).  [oai_citation:3‡file.pdf](sediment://file_00000000636071fd9f0b1bf5c940c7f9)

## Repository Structure

```text
.
├── phia_demo.ipynb
├── phia_agent.py
├── prompt_templates.py
├── glucose_data_utils.py
├── glucose_rules.py
├── sample_cgm.csv
├── sample_meals.csv
├── sample_profile.csv
├── requirement_glucose.txt
└── xt_few_shots/
