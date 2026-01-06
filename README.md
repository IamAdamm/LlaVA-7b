Prompt Sensitivity in Vision–Language Models

A Controlled Spatial Reasoning Study on LLaVA

This repository contains code, prompts, and analysis for a controlled study on prompt sensitivity in Vision–Language Models (VLMs). We evaluate how prompt wording, answer option ordering, and structured reasoning prompts affect spatial reasoning performance in LLaVA on simple spatial relations.

The project focuses on prompting alone — no model fine-tuning, no architectural changes.


What This Project Shows
	•	LLaVA’s spatial reasoning is highly sensitive to prompt design
	•	Shuffling multiple-choice options can push accuracy below random chance
	•	High confidence does not imply correct visual grounding
	•	Structured reasoning prompts (CoT, Scene-Graph CoT) significantly improve accuracy
	•	Scene-Graph CoT performs best by explicitly separating perception and reasoning


Task

Spatial relations (4-way classification):
	•	Left
	•	Right
	•	On top
	•	Under

Dataset:
	•	WhatsUp (Control Group 2)
	•	418 controlled images
	•	Balanced labels (~25% per relation)
  

Prompt Types Evaluated

Multiple Choice (Baseline)
	•	Standard WhatsUp-style prompts
	•	Variants with:
	•	Shuffled answer options
	•	Alternative wording (above / under / below)
	•	Combined shuffling + wording

Chain-of-Thought (CoT)
	•	Model reasons step-by-step before giving a final answer
	•	Final output mapped to one of the four spatial relations

Scene-Graph Chain-of-Thought
	•	Two-stage reasoning:
	1.	Explicit object and relation description (scene graph)
	2.	Final spatial inference from the graph
	•	Strongly constrains reasoning to visual relations


Key Results (Summary)

Prompt Type	Accuracy	Confidence
Multiple Choice	59.81%	48.68%
Chain-of-Thought	66.03%	68.95%
Scene-Graph CoT	70.10%	60.99%

Multiple-choice prompts show strong ordering and wording biases, while structured reasoning significantly improves robustness and accuracy.


Repository Structure

.
├── prompts/
│   └── prompts.json
├── results/
│   └── results.json
├── scripts/
│   ├── run_experiment.py
│   ├── answer_extraction.py
│   └── results_analysis.py
├── figures/
│   ├── MultipleChoiceAccuracy.png
│   ├── MultipleChoiceAccuracyPerGroundTruthOption.png
│   ├── AllPromptsAccuracy.png
│   └── AllPromptsAccuracyPerGroundTruthOption.png
├── notebook.md
└── README.md



Running the Code
	1.	Set up LLaVA and required checkpoints
	2.	Install dependencies
	3.	Run experiments:

python scripts/run_experiment.py


	4.	Analyze results:

python scripts/results_analysis.py



Experiments log results locally using Weights & Biases (offline mode).


Why This Matters

These experiments show that many VLM failures in spatial reasoning are not due to missing visual information, but to prompt-induced heuristic shortcuts. Small prompt changes can completely alter model behavior, confidence, and failure modes.

Structured prompts that explicitly guide perception and reasoning significantly reduce these effects.


Limitations
	•	Controlled images only (WhatsUp subset)
	•	Simple spatial relations
	•	Single model (LLaVA)
	•	No architectural or attention-level intervention


Author

Adam Astamir
University of Hamburg

Just say the word.
