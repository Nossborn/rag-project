import re
import spacy

from dataset import StudieinfoDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS


class RAG:

    RETRIVER_NAME = "sentence-transformers/all-MiniLM-l6-v2"

    def __init__(self, dataset: StudieinfoDataset, k: int):
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=self.RETRIVER_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        db = FAISS.from_documents(dataset, embeddings)

        sk = {'k': k, 'fetch_k': k * 10, 'lambda_mult': 0.10}
        self.retriever = db.as_retriever(search_kwargs=sk)

    def run(self, question: str):
        return [StudieinfoDataset.prepend_metadata(doc)
                for doc in self.retriever.invoke(question)]


class TFIDF:

    def __init__(self, dataset: StudieinfoDataset, k: int):
        self.dataset = dataset

        self.vectorizer = TfidfVectorizer()
        nlp = spacy.load('en_core_web_sm')
        data = (self.preprocess(doc.page_content, nlp) for doc in self.dataset)
        self.docs = self.vectorizer.fit_transform(data)
        self.k = k

    def preprocess(self, text: str, nlp: spacy.language.Language):
        pre = " ".join([token.lemma_.lower() for token in nlp(text)
                        if (not token.is_stop) and (not token.is_punct)])
        pre = re.sub(" +", " ", pre)
        pre = re.sub("\n ", "\n", pre)
        return pre

    def run(self, question: str):
        q_vec = self.vectorizer.transform([question])
        res = cosine_similarity(self.docs, q_vec)
        indicies = res.argpartition(-self.k, axis=0)[-self.k:].flatten()
        return [self.dataset[i] for i in indicies]
