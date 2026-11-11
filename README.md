# 📡 Airtel vs Jio: Competitive Intelligence Dashboard

This project is an end-to-end, fully offline Python application that analyzes telecom network data to compare Airtel and Jio. It features an interactive dashboard built with Streamlit and uses `llama-cpp-python` to generate automated, territory-specific SWOT (Strengths, Weaknesses, Opportunities, Threats) analyses entirely on your local machine.

## ✨ Features
*   **100% Offline LLM:** Generates strategic insights without sending sensitive network data to external APIs like OpenAI.
*   **Geospatial Mapping:** Visualizes site locations and highlights vulnerable zones.
*   **Hierarchical Filtering:** Drill down from State -> District -> Tehsil -> Town.
*   **Automated Metrics:** Calculates Payload Disparity, Proximity Threat Levels, and 5G Vulnerability on the fly.

## 🛠️ Prerequisites

1. **Python:** Ensure you have Python 3.8 or higher installed.
2. **C++ Compiler (Optional but recommended):** Required if you want `llama-cpp-python` to compile with hardware acceleration (like CUDA or Metal) for faster LLM inference.

## 🚀 Installation & Setup

**1. Install Required Python Libraries**
Open your terminal or command prompt and run:
```bash
pip install pandas numpy streamlit llama-cpp-python
```

## Alternate option to install llama-cpp-python if 'long-path extraction error' is encountered
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**2. Download a Local LLM (.gguf format)**
You need a quantized model file to run the AI analysis locally. 
* Go to [Hugging Face](https://huggingface.co/TheBloke) (e.g., search for Mistral-7B-Instruct-v0.2-GGUF).
* Download a model file like `mistral-7b-instruct-v0.2.Q4_K_M.gguf`.
* Place this `.gguf` file in the exact same folder as your Python script (`app.py`).
* *Note: Update the `MODEL_PATH` variable in `app.py` to match the exact name of the file you downloaded.*

**3. Prepare Your Data**
* Ensure your dataset is saved as `input.csv` in the same directory as your script.
* The CSV must contain the columns specified in your schema (e.g., `State`, `District Name`, `Town Name`, `ASiteID Total PL in GB`, `JSiteID Total PL in GB`, `Latitude`, `Longitude`, etc.).

**4. Run the Dashboard**
Execute the following command in your terminal:
```bash
streamlit run app.py
```
or
```bash
python -m streamlit run app.py
```

This will open a local web server and launch the dashboard in your default web browser (usually at `http://localhost:8501`).
Local URL: http://localhost:8501
Network URL: http://10.139.13.212:8501

## 📊 Usage Instructions
1. Use the **Sidebar** to filter data down to a specific Town.
2. Review the top KPIs (Average Payloads, 5G Penetration, and High-Risk Sites).
3. Check the **Geospatial View** to see site distributions.
4. Click **"Generate SWOT Insights"** in the bottom right panel to trigger the local LLM. It will analyze the on-screen metrics and generate a concise SWOT report.

## ⚠️ Troubleshooting
* **Model fails to load:** Ensure the `MODEL_PATH` string exactly matches your downloaded `.gguf` file name and location.
* **LLM is very slow:** If generating the SWOT takes more than a minute, your CPU might be struggling. Try downloading a smaller model (e.g., a `Q3` quantization instead of `Q4`, or a smaller 3B parameter model).
* **Missing Columns:** If the app crashes on load, verify your `input.csv` headers exactly match the column names referenced in the script.