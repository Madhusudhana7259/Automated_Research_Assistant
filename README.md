## 🧠 Autonomous Research Lab Agent

This project implements an **Autonomous Research Lab Agent** using **LangGraph** and **ChatGroq** to automate end-to-end scientific research tasks. The system generates research ideas, retrieves papers, summarizes literature, critiques ideas, designs experiments, and produces a final PDF report.

---

### 🚀 Features

✅ **Topic Generator** – Automatically generates relevant and novel research topics.
✅ **Paper Retriever** – Finds and fetches academic papers (via arXiv).
✅ **Summarizer** – Summarizes key papers using FAISS + HuggingFace embeddings.
✅ **Idea Synthesizer** – Proposes innovative research ideas.
✅ **Critic Agent** – Critically analyzes proposed ideas for feasibility and originality.
✅ **Experiment Designer** – Outlines detailed experimental plans.
✅ **Report Generator** – Produces a well-formatted PDF report (no local storage, directly downloadable).

---

### 🛠 Tech Stack

* **Python**
* **LangGraph**
* **ChatGroq (Qwen-32B)** or equivalent LLM
* **arXivLoader**
* **PyPDFDirectoryLoader**
* **FAISS + HuggingFace Embeddings**
* **FPDF** (for PDF generation)
* **Streamlit** (optional UI)


---

### ⚙️ Installation

```bash
git clone https://github.com/your-username/autonomous-research-lab-agent.git
cd autonomous-research-lab-agent
pip install -r requirements.txt
```

---

### 🚀 Run the project

```bash
python app.py
```

*or with Streamlit*

```bash
streamlit run app.py
```

---

### 📄 Output

✅ A **PDF report** containing:

* Summary
* Research ideas
* Critique
* Detailed experimental plan

The report is generated in memory and offered for download (no local storage required).

---

### 💡 Example Use Cases

* Assisting researchers in rapidly prototyping research directions
* Supporting students in writing literature reviews or experiment designs
* Exploring novel AI-generated research avenues

---

### 📌 TODO / Enhancements

* [ ] Add citation support in the report
* [ ] Integrate more paper sources (e.g., Semantic Scholar)
* [ ] Add charts/diagrams to the PDF
* [ ] Improve Streamlit UI

---

### 📝 License

This project is licensed under the **MIT License**.

---

### 🙌 Acknowledgements

* [LangGraph](https://langgraph.org/)
* [arXiv](https://arxiv.org/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [HuggingFace](https://huggingface.co/)

