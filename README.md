# Personal Glucose Health Insight Agent

A research prototype for **agentic reasoning over glucose time-series data**, combining a **PHIA-style Thought–Act–Observe workflow** with **CGM-specific preprocessing** for personalized, explainable, and **non-diagnostic** glucose insights.

---

## Why this project?

Large language models are increasingly being used to reason over personal health data, but most current systems still struggle with **time-series-specific numerical reasoning**, **event-centered interpretation**, and **domain-grounded explanation**.

This project explores a simple but useful question:

> Can an agent framework originally designed for wearable health insights be adapted to **continuous glucose monitoring (CGM)** data in a way that preserves temporal structure and supports more personalized glucose reasoning?

This repository is an **early research prototype** built around that question.

---

## Current status

This prototype already supports:

- CGM-oriented preprocessing
- meal-centered glucose window construction
- daily glucose summary generation
- lightweight rule-based interpretation
- PHIA-style agent reasoning over structured glucose tables
- natural-language answering for non-diagnostic glucose questions

The current focus is **not** clinical deployment.  
The current focus is to build a **research-ready prototype** that can be used to study:

- event-centered time-series reasoning
- agent-based analysis of physiological signals
- glucose-specific interpretation quality
- benchmark-guided evaluation and error analysis

---

## Project motivation

This project is motivated by two complementary lines of work.

### 1) Agentic personal health reasoning

The first motivation comes from the **Personal Health Insights Agent (PHIA)** line of work: an agent framework that uses iterative reasoning, code generation, and retrieval to analyze personal health data.

PHIA is interesting because it treats personal health analysis as an **open-ended reasoning problem** rather than only a fixed classification task.  
This repository adopts that general idea and asks:

> What happens if we move that agent framework from general wearable data into the glucose domain?

### 2) Glucose as time-series, not just static values

The second motivation comes from the **glucotype** perspective on CGM analysis.

That line of work emphasizes that glucose dysregulation should not be understood only through:

- single measurements
- coarse daily averages
- or static glycemic summaries

Instead, CGM reveals:

- temporal variability
- postprandial excursions
- individualized response patterns
- local fluctuation dynamics

This repository does **not** attempt to reproduce the original glucotype clustering pipeline in full.  
Instead, it borrows the **time-series viewpoint** and adapts it into a lighter, more interpretable preprocessing layer for downstream agent reasoning.

---

## Research framing

This repository should be understood as:

- **not** a medical device
- **not** a diagnostic system
- **not** a full reproduction of the PHIA or glucotype papers
- **not** a final production system

Instead, it is:

- a **glucose-domain extension** of a PHIA-style agent framework
- a **time-series-aware CGM reasoning prototype**
- an experiment in combining:
  - agentic reasoning
  - physiological time-series preprocessing
  - personalized, evidence-grounded explanation

---

## Core idea

The central idea is to combine:

1. **a PHIA-inspired agent layer**  
   for iterative reasoning over health-related questions

2. **a glucose-specific data processing layer**  
   for converting raw CGM and meal records into interpretable intermediate tables

3. **a lightweight interpretation layer**  
   for making responses more structured, personalized, and explainable

So instead of letting an LLM reason directly over raw glucose traces in an unconstrained way, the system first builds a more structured view of the data and then lets the agent reason over that evidence.

---

## What this system is designed to do

The current prototype is designed to support questions such as:

- What happened to my glucose after this meal?
- Which meals were associated with larger spikes?
- How stable was my glucose across the day?
- Are there recurring high-variability periods?
- How does today compare with previous days?
- What are some non-diagnostic insights I can take away from these patterns?

The goal is **not** to produce clinical diagnosis.  
The goal is to produce **personalized, data-grounded, and interpretable glucose insights**.

---

## Method overview

The repository currently has three main layers.

### 1. Glucose data preprocessing

Raw input tables such as:

- CGM readings
- meal records
- user profile information

are transformed into structured intermediate data representations for downstream reasoning.

This layer is implemented mainly in:

- `glucose_data_utils.py`

Typical outputs include:

- cleaned CGM tables
- meal-centered windows
- day-level summaries
- interpretable glucose-derived features

---

### 2. Lightweight glucose reasoning layer

Rather than reproducing the full glucotype clustering framework, this project uses a lighter strategy centered on:

- event-centered meal windows
- daily aggregation
- spike and recovery features
- rule-based glucose interpretation

This design choice is intentional.

The purpose of this project is not to build the most complex glucose model possible.  
The purpose is to create a **simple, explainable, and agent-compatible intermediate representation** that preserves temporal structure while remaining easy to inspect.

This layer is implemented mainly in:

- `glucose_rules.py`
- parts of `glucose_data_utils.py`

---

### 3. Agent-based reasoning

On top of the structured glucose tables, a PHIA-style agent answers natural-language questions using an iterative workflow:

- **Thought**
- **Act**
- **Observe**

The agent can:

- inspect processed glucose tables
- execute analysis code
- reason step by step over evidence
- optionally use external retrieval for contextual information
- generate personalized but non-diagnostic responses

This layer is implemented mainly in:

- `phia_agent.py`
- `prompt_templates.py`
- `xt_few_shots/`

---

## Design principles

This project is built around the following principles:

### 1. Agent-first reasoning
Preserve the PHIA-style iterative reasoning workflow instead of reducing everything to a one-shot prompt.

### 2. Time-series awareness
Treat CGM as **temporal data**, not only as static summary statistics.

### 3. Event-centered interpretation
Use meal-centered or event-centered windows to capture local glucose dynamics.

### 4. Structured evidence before free-form reasoning
Prefer interpretable intermediate tables and features before asking the agent to generate answers.

### 5. Personalization through user data
Ground responses in the user’s own glucose patterns rather than generic advice alone.

### 6. Non-diagnostic scope
Keep outputs informative and actionable without claiming diagnosis or medical decision authority.

---

## Key features

- CGM-centered preprocessing pipeline
- meal-level glucose feature extraction
- daily glucose summary generation
- lightweight glucose rule engine
- PHIA-style ReAct agent for glucose question answering
- personalized, explainable, non-diagnostic responses
- optional contextual retrieval for background information
- modular code structure for research iteration

---

## What makes this different from a generic LLM demo?

This project is **not** just a chatbot wrapped around CSV files.

Its main difference is that it tries to preserve a **time-series reasoning perspective**:

- glucose is treated as a dynamic signal
- meal events are treated as anchors for local analysis
- intermediate glucose tables are created before free-form language generation
- the agent reasons over evidence rather than improvising unsupported advice

In other words, the emphasis is on **reasoning over structured temporal evidence**, not merely summarizing health data.

---

## Limitations

This project currently has several important limitations.

### 1. Not a clinical system

It does not diagnose diabetes, prediabetes, or any medical condition.

### 2. Lightweight processing only

It does not implement the full glucotype clustering methodology or advanced physiological modeling.

### 3. Early-stage evaluation

The current prototype focuses on functionality and interpretability; evaluation is still being refined.

### 4. Limited scope

The present system is centered on **CGM + meal-linked reasoning** and does not yet integrate broader multimodal data such as:

- activity
- heart rate
- sleep
- medication logs
- lab tests
- clinical notes

### 5. Prototype-level robustness

The current version should be understood as a research prototype rather than a hardened production system.

---

## Planned next steps

The next phase of this project will likely focus on:

- improving reasoning quality evaluation
- adding benchmark-guided analysis of output errors
- refining event-centered glucose windowing
- strengthening temporal reasoning on longer CGM histories
- exploring modular extensions toward richer multimodal health data
- comparing agent-based glucose reasoning with stronger task-specific baselines

---

## What feedback would be especially useful?

If you are looking at this repository as a potential collaborator, mentor, or reader, the most useful feedback would be:

- Is the current framing research-worthy?
- Is the glucose preprocessing layer a reasonable first step?
- Should the next phase emphasize:
  - stronger evaluation,
  - stronger modeling,
  - or broader multimodal integration?
- What would make this prototype more convincing as a research system?

---

## References

1. Merrill, M. A., Paruchuri, A., Rezaei, N., Kovacs, G., Perez, J., Liu, Y., Schenck, E., Hammerquist, N., Sunshine, J., Tailor, S., et al.  
   **Transforming wearable data into personal health insights using large language model agents.**  
   *Nature Communications* (2026).

2. Hall, H., Perelman, D., Breschi, A., Limcaoco, P., Kellogg, R., McLaughlin, T., Snyder, M., et al.  
   **Glucotypes reveal new patterns of glucose dysregulation.**  
   *PLOS Biology* 16(7): e2005143 (2018).

---

## Disclaimer

This repository is for **research and educational purposes only**.

It does **not** provide medical diagnosis, treatment, or clinical decision support.  
Any outputs generated by this system should be treated as **non-diagnostic exploratory insights** and **not** as medical advice.


## Repository structure

```text
.
├── README.md
├── phia_agent.py
├── prompt_templates.py
├── glucose_data_utils.py
├── glucose_rules.py
├── phia_demo_github.ipynb
├── requirement_glucose.txt
├── sample_cgm.csv
├── sample_meals.csv
├── sample_profile.csv
└── xt_few_shots/

