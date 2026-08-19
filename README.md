# Job hunt knowledge bank chatbot

A small RAG (Retrieval-Augmented Generation) agent that answers questions
about job-hunting experiences students have shared, and cites which file
each answer came from.

**Architecture:** raw `.md`/`.docx` files -> chunk & embed -> Chroma vector
store -> Gemma 4 tool-calling agent -> cited answer in a Streamlit chat UI.

The agent itself (`agent.py`) is built directly with LangGraph — a small
state graph with an LLM node and a tool node that loops until the model
stops requesting tools — rather than the `langchain.agents.create_agent`
convenience wrapper. It's a few more lines but shows the actual mechanism:
`StateGraph` + `MessagesState` + `ToolNode` + `tools_condition`.

## Setup

1. Create a virtual environment and install dependencies (can be skip):

   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a free API key at https://aistudio.google.com/apikey, then:

   ```
   cp env_sample.txt .env
   ```

   and paste your key into `.env` in place of `your_api_key_here`.

3. Build the vector store from the sample files in `data/`:

   ```
   python ingest.py
   ```

4. Launch the chat UI:

   ```
   streamlit run app.py
   ```

Try asking: *"What interview questions came up at Google?"*, *"How did
people feel about their internships?"*, or *"What techniques helped people
prepare for behavioral questions?"*

## Adding your own knowledge

Drop more `.md` or `.docx` files into `data/` — one per experience, with a
descriptive filename, since the filename is what shows up as the citation
— and re-run `python ingest.py` to rebuild the vector store.

## Files

- `data/` — sample job-hunting write-ups (3 included so you can try it
  immediately)
- `ingest.py` — loads files, chunks, embeds, and stores them in Chroma; run
  once, and again whenever `data/` changes
- `tools.py` — the agent's `search_knowledge_base` tool
- `agent.py` — builds the Gemma 4 tool-calling agent
- `app.py` — the Streamlit chat UI
- `chroma_db/` — created by `ingest.py`; the local vector store (gitignored)

## Extending beyond job hunting

The pipeline is domain-agnostic. To add another topic, either mix files
into the same `data/` folder for one shared bank, or duplicate the project
with a different `data/` folder and `COLLECTION_NAME` for a separate one.

## Troubleshooting

- **"No knowledge base found"** in the app — you haven't run `python
  ingest.py` yet, or it failed partway through.
- **Empty/irrelevant answers** — check that `ingest.py` printed a nonzero
  chunk count, and that your `GOOGLE_API_KEY` is valid.
- **Tool never gets called** — some models are inconsistent about
  spontaneous tool use; try lowering `temperature` in `agent.py`, or make
  the system prompt's instruction to call the tool more forceful.
- **`ImportError` involving `langchain.agents`** — this project no longer
  imports from `langchain.agents` at all; `agent.py` builds the graph
  directly with `langgraph`. If you see this, you're probably running
  stale bytecode or an old copy of `agent.py` — re-download the file and
  delete any `__pycache__` folder.
- **`ImportError` involving `langgraph.prebuilt`** — run `pip install -r
  requirements.txt` again to make sure `langgraph>=1.0.0` actually
  installed, or run `pip show langgraph` to check your version.
