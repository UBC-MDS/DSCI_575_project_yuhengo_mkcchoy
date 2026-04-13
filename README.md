# Amazon Product Query Assistant

This project explores how to build a smart product search assistant for Amazon products using different information retrieval methods. It compares BM25 keyword retrieval and semantic search with embeddings on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset, then presents the results through an interactive [Streamlit](https://streamlit.io/) app.

## Project maintainers

- [Mickey Choy](https://github.com/Maple018)
- [Yuheng Ouyang](https://github.com/yhouyang02)

## Repository structure

```
DSCI_575_project_yuhengo_mkcchoy
  ├── app/                     # Streamlit app code
  ├── bm25_index/              # BM25 retriever artifacts
  ├── data/                    # Raw and processed data (downloaded separately)
  ├── notebooks/               # Jupyter notebooks for experimentation
  ├── results/                 # Result discussion and analysis
  ├── semantic_index/          # Semantic retriever artifacts
  ├── src/                     # Source code for data processing and retrieval
  ├── environment.yml          # Conda environment specification
  ├── README.md                # Project overview and instructions
```

## Get started

To run the app locally, follow the following steps:

1. Clone the repository and navigate to the project directory.

    ```bash
    git clone https://github.com/UBC-MDS/DSCI_575_project_yuhengo_mkcchoy.git
    cd DSCI_575_project_yuhengo_mkcchoy
    ```

2. Create and activate the  conda` environment.

    ```bash
    conda env create -f environment.yml
    conda activate dsci-575-mkc-yho
    ```

3. Start the Streamlit app. The app should open in your default web browser at `http://localhost:8501`. If it does not open automatically, you can navigate to that URL manually. It should take a few seconds to load the full app.

    ```bash
    streamlit run app/app.py
    ```

4. Enter product-related queries in the input box and click the "Search" button. The results may be limited since our test models are built on a subset of the full dataset. For better results, you can try queries related to the "Appliances" category, such as "quiet dishwasher stainless steel".

5. To stop the app, press `Ctrl+C` in the terminal where the Streamlit app is running.
