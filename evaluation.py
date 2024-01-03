import os
import csv
import json

from time import time
from argparse import ArgumentParser

from dataset import StudieinfoDataset
from retrivers import TFIDF, RAG
from answer_model import AnswerModel


QUESTIONS_PATH = "./evaluation/questions.json"
RESPONSE_PATH = "./evaluation/output.csv"
EVALUATION_PATH = "./evaluation/evaluation.csv"


COLUMNS = ["question", "human answer", "method",
           "model answer", "context", "exec time"]


def pipeline(question: str, answer: str, model: AnswerModel, method: str):
    s_time = time()

    in_prompt, output = model.run(question)

    evaluation = {
        "question": question,
        "human answer": answer,
        "method": method,
        "model answer": output,
        "context": in_prompt,
        "exec time": f"{(time()-s_time)*1000:.2f}"
    }

    with open(RESPONSE_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(evaluation.values())


def generate_responses():
    # Load questions
    questions: list[dict[str, str]] = []
    with open(QUESTIONS_PATH, 'r') as file:
        questions = json.load(file)

    dataset = StudieinfoDataset(path="./dataset/courses")

    # Run pipeline for TFIDF
    retriver = TFIDF(dataset, k=3)
    model = AnswerModel(retriever=retriver)
    for item in questions:
        pipeline(item["question"], item["answer"], model, "TFIDF")

    # Run pipeline for RAG
    retriver = RAG(dataset, k=3)
    model = AnswerModel(retriever=retriver)
    for item in questions:
        pipeline(item["question"], item["answer"], model, "RAG")


def evaluate_question(question: str, answer: str, context: str):
    relevance, faithful, quality = 0, 0, 0

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Question:\n{question}")
    print("-" * 125)
    print(f"Context:\n{context}")
    print("-" * 125)
    print(f"Expected answer:\n{answer}")
    print("-" * 125)
    print(f"Model answer:\n{context}")

    relevance = input("Enter relevance of context to the question: ")
    faithful = input("Enter faithfulness of answer to the context: ")
    relevance = input("Enter quality of answer to the question: ")

    return relevance, faithful, quality


def evaluate_reponses():
    # Load model responses
    responses: list[dict[str, str]] = []
    with open(QUESTIONS_PATH, 'r', newline='') as file:
        reader = csv.DictReader(file)
        responses = [row for row in reader]
    
    print(responses)

    if len(responses) == 0:
        raise ValueError("No generated responses found!")

    for response in responses:
        relevance, faithful, quality = evaluate_question(response["question"],
                                                         response["answer"],
                                                         response["context"])

        evaluation = {**response,
                      "context relevance": relevance,
                      "answer faithfulness": faithful,
                      "answer quality": quality,
                      }

        with open(EVALUATION_PATH, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(evaluation.values())

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
