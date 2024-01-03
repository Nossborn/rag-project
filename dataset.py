import re

from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

            metadata = {"name": name.group(1), "code": code.group(1)}
            document = Document(page_content=document, metadata=metadata)
            self.documents.append(document)

        # Seperates on new course info section
        seperator = r"\n\n(?=[^\n:]+:\n)"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                       chunk_overlap=250,
                                                       separators=[seperator],
                                                       is_separator_regex=True)
        self.documents = text_splitter.split_documents(self.documents)

        print(f"Loaded {len(self.documents)} documents.")

    def __getitem__(self, i):
        doc = self.documents[i]
        return StudieinfoDataset.prepend_metadata(doc)

    def prepend_metadata(document: Document):
        name, code = document.metadata['name'], document.metadata['code']
        return f"Course: {name} ({code})\n" + document.page_content

    def __iter__(self):
        for doc in self.documents:
            yield doc
