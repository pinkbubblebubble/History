# History Deep Research

## Setup

1. Clone the repository:
```bash
git clone https://github.com/CharlesQ9/HistoryDeepResearch.git
cd HistoryDeepResearch
```

2. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On macOS
```

3. Install dependencies:
```bash
pip install -e smolagents/  # Must install the dev version of smolagents!!
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```plaintext
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
SERPAPI_API_KEY=your_serpapi_api_key
SERPER_API_KEY=your_serper_api_key
```

5. Run the application:
```bash
cd smolagents/examples/open_deep_research
streamlit run myApp.py
```

## Project Structure

```
historyWebApp/
├── smolagents/
│   ├── src/
│   │   └── smolagents/
│   └── examples/
│       └── open_deep_research/
│           ├── scripts/
│           ├── myApp.py
│           └── run.py
├── README.md
└── requirements.txt
```


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
