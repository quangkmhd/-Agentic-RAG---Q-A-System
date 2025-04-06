# 🤖 Agentic RAG Evaluation on HuggingFace Documentation

## 🧠 Introduction
Dự án này đánh giá khả năng trả lời câu hỏi của hai hệ thống **Retrieval-Augmented Generation (RAG)** – một hệ thống sử dụng agent với công cụ retriever thông minh (*Agentic RAG*) và một hệ thống sử dụng context trực tiếp từ truy xuất (*Standard RAG*). Cả hai hệ thống được huấn luyện và kiểm thử trên tập dữ liệu tài liệu chính thức của HuggingFace (`huggingface_doc`) và tập câu hỏi kiểm thử `huggingface_doc_qa_eval`.

## 📚 Table of Contents
* [Introduction](#-introduction)
* [Installation](#-installation)
* [Usage](#-usage)
* [Pipeline Overview](#-pipeline-overview)
* [Features](#-features)
* [Evaluation](#-evaluation)
* [Dependencies](#-dependencies)
* [License](#-license)

## 💻 Installation

```bash
pip install pandas langchain langchain-community sentence-transformers faiss-cpu smolagents --upgrade
```

Đăng nhập Hugging Face:

```python
from huggingface_hub import notebook_login
notebook_login()
```

## 🚀 Usage
1. **Tải dữ liệu kiến thức và chia nhỏ thành chunks**
2. **Tạo FAISS vector store và huấn luyện với embedding**
3. **Xây dựng** `RetrieverTool`
4. **Tạo Agent sử dụng** `ToolCallingAgent` và so sánh với RAG tiêu chuẩn
5. **Chạy đánh giá tự động bằng LLM**

## 🔁 Pipeline Overview

### 🗃️ 1. Dataset Loading
```python
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
```

### 📑 2. Text Preprocessing & Embedding
* Tách văn bản bằng `RecursiveCharacterTextSplitter`
* Embedding sử dụng `thenlper/gte-small`
* Tạo FAISS vector store với `DistanceStrategy.COSINE`

### 🧰 3. RetrieverTool
```python
class RetrieverTool(Tool):
    ...
    def forward(self, query: str) -> str:
        docs = self.vectordb.similarity_search(query, k=7)
```

### 🧠 4. ToolCalling Agent
```python
agent = ToolCallingAgent(tools=[retriever_tool], model=HfApiModel("meta-llama/Llama-3.1-70B-Instruct"))
```

### 📥 5. RAG Pipeline
* **Agentic RAG**: Agent sử dụng nhiều truy vấn semantic khác nhau để truy xuất thông tin
* **Standard RAG**: Tách riêng phần truy vấn và cung cấp context trực tiếp cho LLM

### ✅ 6. Evaluation
* Mỗi câu trả lời được đánh giá bởi LLM theo thang điểm từ 1 đến 3
* Mẫu đánh giá:
```
Score Rubrics:
1: Không đúng, không đầy đủ.
2: Đúng một phần.
3: Hoàn toàn đúng, đầy đủ.
```

## ✨ Features
* Semantic RAG với công cụ truy xuất có thể lặp lại truy vấn
* Sử dụng FAISS cho tìm kiếm nhanh chóng
* Đánh giá kết quả với LLM-based judge
* Tự động xử lý toàn bộ pipeline từ huấn luyện đến đánh giá

## 📊 Evaluation
Sau khi chạy, kết quả sẽ hiển thị:
```
Average score for agentic RAG: 91.3%
Average score for standard RAG: 78.5%
```
Điều này cho thấy Agentic RAG có thể cải thiện đáng kể độ chính xác và đầy đủ của câu trả lời.

## 📦 Dependencies
* `pandas`
* `langchain`, `langchain-community`
* `sentence-transformers`
* `faiss-cpu`
* `smolagents`
* `datasets`, `tqdm`
* `transformers`
* `huggingface_hub`

## 📜 License
MIT License © 2025
