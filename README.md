# CoT Batch Generator Pro

**A powerful, real-time Chain-of-Thought dataset generator with a beautiful browser UI.**

Generate high-quality synthetic training data for LLMs with structured Cold Start + Hot Start reasoning — perfect for creating [wop/XXXXXL-style CoT datasets](https://huggingface.co/datasets/wop/XXXXXL-chain-of-thought).

![Demo Screenshot](https://github.com/user-attachments/assets/bea20d6c-0d7e-4da1-8fa0-a2b7b48641ea)

---

## ✨ Features

- **Modern Web UI** – Clean, dark-themed interface with live streaming
- **True Parallel Generation** – Up to 6 concurrent workers using `ThreadPoolExecutor`
- **Real SSE Streaming** – Watch tokens appear in real time
- **Live Statistics** – Progress, speed (chars/sec), ETA, success rate
- **Smart Estimator** – Accurate remaining time prediction
- **Robust Error Handling** – Graceful recovery and detailed logging
- **JSONL Export** – Ready-to-use `train.jsonl` with optional ASCII cleaning
- **One-Click Actions** – Copy JSONL, download, refresh, copy streams
- **Safe Output** – Automatic punctuation normalization & ASCII conversion

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install requests
```

### 2. Configure the Script

Edit the top section of `cot_generator.py`:

```python
API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # Your local LLM server
MODEL = "qwen/qwen3.5-9b"                               # Change to your model
PORT = 8080
MAX_WORKERS = 6                                         # Adjust based on your hardware
```

### 3. Run the Generator

```bash
python cot_generator.py
```

Then open your browser and go to: **http://localhost:8080**

---

## 📋 How to Use

1. Enter your questions (one per line) in the input box
2. Click **▶ Generate Batch**
3. Watch live generation in the stream panel
4. Use the buttons to:
   - Copy JSONL (cleaned & validated)
   - Download `train.jsonl`
   - Refresh loaded entries

---

## Configuration Options

| Variable          | Description                              | Default             |
|-------------------|------------------------------------------|---------------------|
| `API_URL`         | Local LLM API endpoint                   | `http://127.0.0.1:1234/...` |
| `MODEL`           | Model name to use                        | `qwen/qwen3.5-9b`   |
| `MAX_WORKERS`     | Number of parallel generations           | `6`                 |
| `DATA_FILE`       | Output JSONL file                        | `train.jsonl`       |

---

## System Prompt

The generator uses a carefully designed system prompt that forces the model to output structured thinking with:

- **Cold Start** – Initial analysis and context awareness
- **Hot Start** – Step-by-step reasoning
- `<think>...</think>` tags (as requested)

You can easily customize the `SYSTEM_PROMPT` variable in the script.

---

## Output Format

Each entry follows this structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Enable thinking features: INTUITION, COLD START, HOT START"
    },
    {
      "role": "user",
      "content": "How many R in strawberry"
    },
    {
      "role": "assistant",
      "content": "<think>\n### Cold start\n...\n</think>\n\n**Final Answer**"
    }
  ]
}
```

---

## Tips for Best Results

- Use a strong reasoning model (Qwen2.5, Llama-3.1/3.3, etc.)
- Keep temperature around **0.7**
- Start with 5–20 questions to test
- For very large batches, increase `MAX_WORKERS` (if your hardware allows)

---

## Project Structure

```
.
├── cot_generator.py          # Main script (self-contained)
├── train.jsonl               # Generated dataset (auto-created)
└── README.md
```

---

## Contributing

Feel free to open issues or PRs! Especially welcome:
- Better UI components
- Support for OpenAI/Anthropic APIs
- Prompt engineering improvements
- Export format options

---

**Made for high-quality synthetic CoT data generation.**

*Happy dataset building! 🧠*
