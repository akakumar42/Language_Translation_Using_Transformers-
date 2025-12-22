# PyTorch Transformer for Machine Translation

A complete implementation of the Transformer architecture for neural machine translation from English to German, based on the "Attention is All You Need" paper by Vaswani et al.

## Features

- **Full Transformer Implementation**: Encoder-decoder architecture with multi-head self-attention and cross-attention mechanisms
- **English to Deutsch Translation**: Trained on the OPUS Books dataset
- **Decoding Strategies**: Greedy decoding and beam search for inference
- **Attention Visualization**: Interactive visualizations of attention weights using Altair
- **Flexible Training**: Support for training on Google Colab and local machines
- **Experiment Tracking**: Integration with Weights & Biases for logging and monitoring
- **Custom Tokenization**: Word-level tokenization using Hugging Face tokenizers

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pytorch-transformer
```

2. Create a conda environment (optional but recommended):
```bash
conda env create -f conda.txt
conda activate transformer-env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

#### Local Training
Run the training script locally:
```bash
python train.py
```

#### Colab Training
Use the provided Colab notebook `Colab_Train.ipynb` for training on Google Colab with GPU support.

#### Weights & Biases Training
For experiment tracking:
```bash
python train_wb.py
```

### Inference

#### Translate Text
Use the translation script to translate English sentences to Italian:
```bash
python translate.py "Hello, how are you?"
```

#### Interactive Inference
Use the `Inference.ipynb` notebook for interactive translation and exploration.

#### Beam Search
Explore beam search decoding in `Beam_Search.ipynb`.

### Visualization

Run `attention_visual.ipynb` to visualize attention mechanisms in the trained model.

## Project Structure

- `model.py`: Transformer model architecture implementation
- `train.py`: Main training script
- `train_wb.py`: Training with Weights & Biases logging
- `translate.py`: Command-line translation tool
- `config.py`: Configuration parameters
- `dataset.py`: Data loading and preprocessing
- `Colab_Train.ipynb`: Google Colab training notebook
- `Local_Train.ipynb`: Local training notebook
- `Inference.ipynb`: Interactive inference notebook
- `Beam_Search.ipynb`: Beam search implementation
- `attention_visual.ipynb`: Attention visualization notebook
- `requirements.txt`: Python dependencies
- `conda.txt`: Conda environment specification

## Configuration

Key parameters in `config.py`:
- `batch_size`: 8
- `num_epochs`: 20
- `lr`: 1e-4
- `seq_len`: 350
- `d_model`: 512
- `datasource`: 'opus_books'
- `lang_src`: "en"
- `lang_tgt`: "it"

## Requirements

- Python 3.9+
- PyTorch 2.0.1
- Hugging Face datasets and tokenizers
- Altair for visualization
- Weights & Biases for experiment tracking
- Other dependencies listed in `requirements.txt`

## Model Architecture

The implementation includes:
- Multi-head attention (8 heads)
- 6 encoder and 6 decoder layers
- Feed-forward networks with 2048 hidden units
- Layer normalization and residual connections
- Xavier uniform parameter initialization

## Dataset

Trained on the OPUS Books dataset, which contains parallel English-Italian text from books.

## License

MIT License

