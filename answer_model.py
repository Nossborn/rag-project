import re

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from auto_gptq import exllama_set_max_input_length

from retrivers import TFIDF, RAG


class AnswerModel:

    LM_NAME = "TheBloke/Mistral-7B-OpenOrca-GPTQ"

    LM_MAX_INPUT_LEN = 4096
    LM_TEMP = 0.7

    TEMPLATE = re.sub(" +", " ",
                      """<|im_start|>system
                         Answer the question based only on the following context:
                         {context}<|im_end|>
                         <|im_start|>user
                         {question}<|im_end|>
                         <|im_start|>assistant
                         """)

    def __init__(self, retriever: TFIDF | RAG):
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_template(self.TEMPLATE)

        model = AutoModelForCausalLM.from_pretrained(self.LM_NAME,
                                                     device_map="cuda",
                                                     trust_remote_code=False,
                                                     revision="main",)
        model = exllama_set_max_input_length(model, self.LM_MAX_INPUT_LEN)
        tokenizer = AutoTokenizer.from_pretrained(self.LM_NAME, use_fast=True)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.LM_TEMP,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1
        )

        self.lm = HuggingFacePipeline(pipeline=pipe)

    def run(self, question: str):
        context = self.retriever.run(question)
        instruction = self.prompt.invoke({
            "context": "\n".join(context),
            "question": question
        })
        output = self.lm.invoke(instruction)
        return instruction.to_string().strip(), output
