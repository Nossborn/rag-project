import re

from pathlib import Path
from langchain_core.documents import Document


class StudieinfoDataset:

    def __init__(self, path: str):
        data_path = Path(path)

        self.documents: list[Document] = []
        for path in data_path.glob(f"t*.txt"):
            document = None
            with open(path, encoding="utf-8") as f:
                document = "".join(f.readlines())

            code = re.search(r'Course code:\s(.*)', document)
            name = re.search(r'Course title:\s*(.+)', document)
            if code is None or name is None:
                raise Exception(f"No code or name for path: {path}")

            if len(code.group(1)) < 6:
                print(f"Unsual code warning: {code.group(1)} ({path})")

            document = self.preprocess(document)
            metadata = {"name": name.group(1), "code": code.group(1)}
            document = Document(page_content=document, metadata=metadata)
            self.documents.append(document)

        print(f"Loaded {len(self.documents)} documents.")

    def preprocess(self, document: str):
        return document

    def __getitem__(self, i):
        return self.documents[i].page_content

    def __iter__(self):
        for doc in self.documents:
            yield doc
