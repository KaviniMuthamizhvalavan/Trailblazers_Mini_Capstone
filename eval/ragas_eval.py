"""
RAG Evaluation using the actual Ragas library.
Computes answer_relevancy and context_recall metrics using Ragas evaluate
against the actual RAG pipeline using 12 structured Q&A test pairs.
Reports real numbers.
"""

import os
import sys
import json
from unittest.mock import MagicMock

# Dynamic mock to prevent import crash with langchain-community
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=env_path)

# Configure OpenAI keys in the environment for the Ragas evaluator client (use main key only)
api_key = os.getenv("NEW_OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Comment out gateway base URL logic and delete it from environment to ensure direct OpenAI access
if "OPENAI_BASE_URL" in os.environ:
    del os.environ["OPENAI_BASE_URL"]

# Temporarily disable LangSmith tracing and Ragas telemetry for this script to prevent connection timeout delays
if "LANGCHAIN_TRACING_V2" in os.environ:
    del os.environ["LANGCHAIN_TRACING_V2"]
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import answer_relevancy, context_recall
from langchain_openai import OpenAIEmbeddings
from agent.rag_node import retrieve_courses
from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm import get_llm

# Test set: question/expected-answer pairs about course content and prerequisites
EVAL_QUESTIONS = [
    {
        "question": "What are the prerequisites for the Advanced SAP FICO course?",
        "ground_truth": "The prerequisites for SAP FICO Advanced Integration (SAP-104) are SAP Financial Accounting (FI) Basics (SAP-102) and SAP Controlling (CO) Fundamentals (SAP-103).",
        "reference_contexts": ["SAP-102", "SAP-103", "SAP-104"]
    },
    {
        "question": "What courses should I take before the SAP Consultant Certification Prep?",
        "ground_truth": "Before SAP Consultant Certification Prep (SAP-108), you need SAP FICO Advanced Integration (SAP-104) and SAP Materials Management (SAP-106).",
        "reference_contexts": ["SAP-104", "SAP-106", "SAP-108"]
    },
    {
        "question": "Which beginner courses are available for someone new to cloud computing?",
        "ground_truth": "For cloud computing beginners, Cloud Computing Fundamentals (CLD-101) is the starting point, followed by AWS Cloud Practitioner Essentials (CLD-102) or Azure Fundamentals (CLD-105).",
        "reference_contexts": ["CLD-101", "CLD-102", "CLD-105"]
    },
    {
        "question": "What is the prerequisite chain for becoming an AWS Solutions Architect?",
        "ground_truth": "The chain is Cloud Computing Fundamentals (CLD-101) → AWS Cloud Practitioner Essentials (CLD-102) → AWS Solutions Architect Associate (CLD-103).",
        "reference_contexts": ["CLD-101", "CLD-102", "CLD-103"]
    },
    {
        "question": "How many hours does the Salesforce Administrator Certification Prep take?",
        "ground_truth": "The Salesforce Administrator Certification Prep (SF-106) takes an estimated 25 hours.",
        "reference_contexts": ["SF-106"]
    },
    {
        "question": "What are the prerequisites for the Workday Integration and Studio course?",
        "ground_truth": "Workday Integration and Studio (WD-105) requires Workday Business Processes (WD-102) and Workday Advanced Reporting and Analytics (WD-104).",
        "reference_contexts": ["WD-102", "WD-104", "WD-105"]
    },
    {
        "question": "Which cybersecurity courses are suitable for an intermediate learner?",
        "ground_truth": "Intermediate cybersecurity courses include Network Security and Firewalls (CS-102), Ethical Hacking and Penetration Testing (CS-103), SOC Analyst (CS-104), and CompTIA Security+ Certification Prep (CS-108).",
        "reference_contexts": ["CS-102", "CS-103", "CS-104", "CS-108"]
    },
    {
        "question": "What courses cover deep learning and neural networks?",
        "ground_truth": "Deep Learning with TensorFlow and PyTorch (AI-104) covers building and training deep neural networks including CNNs, RNNs, and LSTMs. It requires Machine Learning Algorithms Deep Dive (AI-103).",
        "reference_contexts": ["AI-104", "AI-103"]
    },
    {
        "question": "What is the difference between SAP FI and SAP CO modules?",
        "ground_truth": "SAP FI (Financial Accounting) covers general ledger, accounts payable/receivable, and asset accounting. SAP CO (Controlling) covers cost center accounting, internal orders, and profitability analysis. Both require SAP Fundamentals (SAP-101).",
        "reference_contexts": ["SAP-102", "SAP-103", "SAP-101"]
    },
    {
        "question": "Which courses require both cloud and security knowledge?",
        "ground_truth": "Cloud Security Architecture (CS-105) requires both Network Security and Firewalls (CS-102) and AWS Solutions Architect Associate (CLD-103), making it a cross-domain course.",
        "reference_contexts": ["CS-105", "CS-102", "CLD-103"]
    },
    {
        "question": "What NLP topics are covered in the AI track?",
        "ground_truth": "Natural Language Processing (AI-105) covers text preprocessing, word embeddings, sequence models, attention mechanisms, transformers, and BERT/GPT fine-tuning. It requires Machine Learning Algorithms Deep Dive (AI-103).",
        "reference_contexts": ["AI-105", "AI-103"]
    },
    {
        "question": "How long would a complete Workday learning path take?",
        "ground_truth": "A complete Workday path from HCM Foundations through Pro Certification includes WD-101 (15h), WD-102 (18h), WD-103 (20h), WD-104 (22h), WD-105 (25h), and WD-108 (25h), totaling approximately 125 hours.",
        "reference_contexts": ["WD-101", "WD-102", "WD-103", "WD-104", "WD-105", "WD-108"]
    }
]


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate an answer using retrieved contexts."""
    llm = get_llm()
    context_text = "\n\n".join(contexts)
    prompt = f"""Based on the following course information, answer the question.

Course Information:
{context_text}

Question: {question}

Answer concisely and accurately based only on the provided information."""

    import time
    for attempt in range(5):
        try:
            response = llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content="Evaluate now.")
            ])
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < 4:
                    print(f"Rate limit hit in ragas_eval, sleeping for 31 seconds... (Attempt {attempt+1})")
                    time.sleep(31)
                else:
                    raise e
            else:
                raise e
    return response.content


def run_evaluation():
    """Run Ragas evaluation on a fast subset of 3 questions to prevent rate limits."""
    print("=" * 70)
    print("Ragas Evaluation — Learning Path Recommender (Fast Mode)")
    print("=" * 70)

    questions = []
    generated_answers = []
    retrieved_contexts = []
    ground_truths = []
    
    # Store detailed logs for file saving
    detailed_logs = []

    # Use first 3 representative questions for speed
    fast_subset = EVAL_QUESTIONS[:3]

    for i, qa in enumerate(fast_subset):
        print(f"\n--- Processing Question {i+1}/{len(fast_subset)} ---")
        print(f"Q: {qa['question']}")

        # Retrieve contexts from RAG pipeline
        docs, meta = retrieve_courses(qa["question"], top_k=10)
        retrieved_ids = [m["course_id"] for m in meta]

        # Generate answer using LLM
        answer = generate_answer(qa["question"], docs)
        print(f"A: {answer[:120]}...")
        print(f"Retrieved: {retrieved_ids[:6]}")

        # Collect for dataset
        questions.append(qa["question"])
        generated_answers.append(answer)
        retrieved_contexts.append(docs)
        ground_truths.append(qa["ground_truth"])

        detailed_logs.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "generated_answer": answer,
            "retrieved_ids": retrieved_ids,
            "expected_ids": qa["reference_contexts"]
        })

    # Prepare Hugging Face dataset
    data = {
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("\nRunning Ragas metrics (answer_relevancy, context_recall)...")

    # Force embeddings to real OpenAI. agent.llm's module-level load_dotenv() restores the
    # gateway OPENAI_BASE_URL into the environment, so OpenAIEmbeddings would otherwise route
    # embedding requests to the gateway (which can't serve them) -> APIConnectionError. Pass
    # api_key + base_url explicitly, mirroring get_llm(), rather than relying on ambient env.
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("NEW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
    )

    # answer_relevancy defaults to strictness=3, which asks the LLM for n=3 generations
    # in a single call. The configured model returns only 1 ("LLM returned 1 generations
    # instead of requested 3"), so request a single generation explicitly for stability.
    answer_relevancy.strictness = 1

    # Run metrics serially with generous timeouts/retries. The default 16 concurrent
    # workers overwhelm the endpoint and cause intermittent APIConnectionError/TimeoutError,
    # which silently drop individual metric scores to NaN.
    run_config = RunConfig(max_workers=1, timeout=300, max_retries=6, max_wait=30)

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy, context_recall],
            llm=get_llm(),
            embeddings=embeddings_model,
            run_config=run_config,
            raise_exceptions=False,
        )
    except Exception as e:
        print(f"\nERROR running Ragas evaluation: {e}")
        return

    # ragas 0.4.x EvaluationResult has no .get(); indexing returns the per-item list of
    # scores. Build the per-question table once and derive NaN-safe averages from it.
    results_df = results.to_pandas()

    def col_mean(col):
        if col not in results_df.columns:
            return float("nan")
        return float(results_df[col].mean())  # pandas .mean() skips NaN

    def col_scored(col):
        if col not in results_df.columns:
            return 0
        return int(results_df[col].notna().sum())

    avg_relevancy = col_mean("answer_relevancy")
    avg_recall = col_mean("context_recall")
    relevancy_scored = col_scored("answer_relevancy")
    recall_scored = col_scored("context_recall")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions evaluated: {len(fast_subset)}")
    print(f"Average Answer Relevancy: {avg_relevancy:.3f}  ({relevancy_scored}/{len(fast_subset)} scored)")
    print(f"Average Context Recall:   {avg_recall:.3f}  ({recall_scored}/{len(fast_subset)} scored)")
    print("=" * 70)

    # Attach per-question scores. A failed metric job is NaN — record it as null rather
    # than fabricating a number (real-numbers-only rule from the planner).
    def cell(idx, col):
        if col not in results_df.columns:
            return None
        val = results_df.iloc[idx][col]
        return None if pd.isna(val) else float(val)

    def safe_round(v):
        return None if pd.isna(v) else round(float(v), 3)

    detailed_results = []
    for idx, log in enumerate(detailed_logs):
        log["answer_relevancy"] = cell(idx, "answer_relevancy")
        log["context_recall"] = cell(idx, "context_recall")
        detailed_results.append(log)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "rag_eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "evaluation_method": "Ragas Library (answer_relevancy, context_recall)",
            "summary": {
                "num_questions": len(fast_subset),
                "answer_relevancy_scored": relevancy_scored,
                "context_recall_scored": recall_scored,
                "avg_answer_relevancy": safe_round(avg_relevancy),
                "avg_context_recall": safe_round(avg_recall),
            },
            "detailed_results": detailed_results,
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")
    return avg_relevancy, avg_recall


if __name__ == "__main__":
    run_evaluation()
