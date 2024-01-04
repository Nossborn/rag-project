import os
import json
import pandas as pd

from time import time
from argparse import ArgumentParser

from dataset import StudieinfoDataset
from retrivers import TFIDF, RAG
from answer_model import AnswerModel

from transformers import logging
logging.set_verbosity_error()


QUESTIONS_PATH = "./evaluation/questions.json"
RESPONSE_PATH = "./evaluation/output.csv"
EVALUATION_PATH = "./evaluation/evaluation.csv"


COLUMNS = ["question", "human_answer", "method",
           "model_answer", "context", "exec_time"]

EVAL_COLUMNS = COLUMNS + ["context_relevance", "answer_faithfulness",
                          "answer_quality"]


def pipeline(question: str, answer: str, model: AnswerModel, method: str):
    s_time = time()
    in_prompt, output = model.run(question)
    return {
        "question": question,
        "human_answer": answer,
        "method": method,
        "model_answer": output,
        "context": in_prompt,
        "exec_time": f"{(time()-s_time)*1000:.2f}"
    }


def generate_responses():
    # Load questions
    questions: list[dict[str, str]] = []
    with open(QUESTIONS_PATH, 'r') as file:
        questions = json.load(file)

    dataset = StudieinfoDataset(path="./dataset/courses")

    evaluations = []

    # Run pipeline for TFIDF
    retriver = TFIDF(dataset, k=3)
    model = AnswerModel(retriever=retriver)
    for i, item in enumerate(questions):
        print(f"Processing question {i+1}")
        evaluations.append(pipeline(item["question"], item["answer"],
                                    model, "TFIDF"))

    dataframe = pd.DataFrame(evaluations)
    dataframe.to_csv(RESPONSE_PATH, mode="w")

    # Run pipeline for RAG
    retriver = RAG(dataset, k=3)
    model = AnswerModel(retriever=retriver)
    for i, item in enumerate(questions):
        print(f"Processing question {i+1}")
        evaluations.append(pipeline(item["question"], item["answer"],
                                    model, "RAG"))

    dataframe = pd.DataFrame(evaluations)
    dataframe.to_csv(RESPONSE_PATH, mode="w")


def evaluate_question(question: str, answer: str, context: str, model_answer: str):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Question:\n{question}")
    print("-" * 125)
    print(f"Context:\n{context}")
    print("-" * 125)
    print(f"Expected answer:\n{answer}")
    print("-" * 125)
    print(f"Model answer:\n{model_answer}")
    print("-" * 125)

    relevance = input("Enter relevance of context to the question (1-5): ")
    faithful = input("Enter faithfulness of answer to the context (1-5): ")
    quality = input("Enter quality of answer to the question (1-5): ")

    return relevance, faithful, quality


def evaluate_reponses():
    # Load model responses
    responses = pd.read_csv(RESPONSE_PATH)

    if len(responses) == 0:
        raise ValueError("No generated responses found!")

    evaluations = []
    for row in responses.itertuples(index=False):
        relevance, faithful, quality = evaluate_question(row.question,
                                                         row.human_answer,
                                                         row.context,
                                                         row.model_answer)

        evaluations.append({
            "question": row.question,
            "human_answer": row.human_answer,
            "method": row.method,
            "model_answer": row.model_answer,
            "context": row.context,
            "exec_time": row.exec_time,
            "context_relevance": relevance,
            "answer_faithfulness": faithful,
            "answer_quality": quality,
        })

        dataframe = pd.DataFrame(evaluations)
        dataframe.to_csv(EVALUATION_PATH, mode="w")


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluator of RAG pipeline.')

    parser.add_argument('mode', type=str, help="Select mode, generate" +
                                               " reponses ('gen') or evaluate" +
                                               " generated responses ('eval').")

    args = parser.parse_args()

    mode: str = args.mode

    if mode == "gen":
        generate_responses()

    elif mode == "eval":
        evaluate_reponses()
