# Scamware detection with GAT and LLMs
This repository contains the code and methods described in the paper. The paper presents an analysis of mobile application activities using Graph Attention Networks (GAT) and large language models (LLMs) for classification and fraud detection to check UI-centric fraudulent apps.
The project aims to provide tools for constructing, training, and evaluating graph-based models for mobile app activity analysis, with the added functionality of interacting with language models for deeper semantic analysis and reporting.

### Project Structure
```
/project-root
│
├── /setg                    # Main source code for data processing, model definition, and evaluation
│   ├── gen_setg.py           # Generates graph data for activity analysis
│   ├── gen_setg_test.py      # Generates testing datasets for model evaluation
│   ├── GAT_classifier/       # Contains code for GAT classifier implementation
│   ├── util/                 # Utility functions for data loading, logging, and model pooling
│   ├── model/                # Contains the model definitions (GAT, Autoencoder, etc.)
│   └── llm_connector/        # Connects to LLM APIs like ChatGPT for semantic analysis
├── one-key-run.sh            # Script for running the full pipeline
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

### Requirements
To run this project, make sure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

### How to Reproduce the Results from the Paper

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/project-name.git
    cd project-name
    ```
2. **Install dependencies:**
    Make sure to install all required libraries by running:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the experiment pipeline:**
    The project includes a one-key script to run the full pipeline (data generation, model training, and evaluation). Use the following command:
    ```bash
    bash one-key-run.sh
    ```
    This will automatically:
    - Generate graph data from mobile app activities.
    - Train the GAT model.
    - Evaluate the model performance.
    - Interact with the LLM for activity reports and analysis.

4. **Model Evaluation:**
    After running the pipeline, the results will be saved in the `output/` directory. You can find model performance metrics, graphs, and LLM-generated reports there.
    To manually test the model, you can run:
    ```bash
    python setg/GAT_classifier/test.py
    ```
    This script will load the trained GAT model and evaluate it on the test dataset.
