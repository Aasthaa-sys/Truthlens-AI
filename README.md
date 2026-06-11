* TruthLens AI

Explainable Multi-Agent LLM Framework for Detecting Phishing and Social Engineering Attacks

 Overview

TruthLens AI is an intelligent, explainable multi-agent system designed to detect phishing attempts, social engineering attacks, and suspicious content using Large Language Models (LLMs).

Unlike traditional black-box models, this system provides:
- Transparent reasoning
- Risk scoring
- Multi-step analysis using AI agents

🎯 Problem Statement

Phishing and social engineering attacks are increasing rapidly and are becoming harder to detect using traditional rule-based or ML systems.

Existing systems often:
- Lack explainability
- Fail on contextual attacks
- Do not analyze intent deeply

 Solution

TruthLens AI uses a **multi-agent AI architecture**, where each agent performs a specialized task:

- Intent Analysis Agent
- Phishing Pattern Detection Agent
- Context & Consistency Checker
- Risk Scoring Engine

All outputs are combined to produce a final explainable verdict.

🏗️ Architecture
User Input
↓
Preprocessing Layer
↓
Agent 1 → Intent Analysis
Agent 2 → Phishing Pattern Detection
Agent 3 → Context & Consistency Check
↓
Risk Scoring Engine
↓
Final Verdict + Explanation

⚙️ Tech Stack

- Python
- NLP (Natural Language Processing)
- Large Language Models (LLMs)
- Machine Learning (optional modules)
- Pandas, NumPy
- Scikit-learn (if used)
- Flask / Streamlit (if applicable)

✨ Features

-  Phishing detection using AI agents
-  Explainable AI (transparent decision making)
-  Risk scoring system (0–100 scale)
-  Context consistency checking
-  Modular multi-agent architecture
-  Dataset-based evaluation support

 📁 Project Structure

 Truthlens-AI/
│
├── app/ # Main application code
├── agents/ # AI agent modules
├── models/ # ML / LLM models
├── notebooks/ # Experiments & testing
├── requirements.txt # Dependencies
└── README.md # Project documentation


🛠️ Installation

bash
git clone https://github.com/Aasthaa-sys/Truthlens-AI.git

cd Truthlens-AI

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

How to Run
python app/main.py

Sample Output
Input:
"Your bank account is suspended. Click here immediately to verify."

Output:
Risk Score: 92/100 🔴 HIGH RISK

Explanation:
- Urgency-based manipulation detected
- Suspicious external link patterns
- Matches phishing behavior patterns
- No verified sender identity
🔮 Future Improvements
Browser extension for real-time detection
Email integration system
Multilingual phishing detection
Cloud deployment (AWS / Azure / HuggingFace)
Fine-tuned domain-specific LLM

👨‍💻 Author
Aastha Desai

Internship Project: TruthLens AI
Focus: Explainable AI, NLP, Multi-Agent Systems

⭐ Note

This project demonstrates an AI-based approach to cybersecurity threat detection using explainable multi-agent reasoning.


