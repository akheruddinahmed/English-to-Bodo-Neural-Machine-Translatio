# English to Bodo Neural Machine Translation (NMT)

This project implements a **Neural Machine Translation (NMT)** system that translates English sentences into Bodo using an **attention-based Seq2Seq architecture**. It addresses the challenges of low-resource translation for the Bodo language using modern deep learning techniques.

---

## 📌 Project Overview

- **Language Pair**: English ➝ Bodo  
- **Architecture**: Bidirectional GRU Encoder + GRU Decoder with Bahdanau Attention  
- **Framework**: PyTorch  
- **Corpus Domain**: Tourism and Health  
- **Dataset Size**: ~33,000 sentence pairs  
- **BLEU-1 Score**: 14.64%

---

## 🧠 Model Architecture

The model uses a sequence-to-sequence (Seq2Seq) architecture enhanced by an attention mechanism for effective translation:

- The **encoder** consists of a two-layer bidirectional GRU that processes the English input sentence from both directions to capture rich contextual information.
- The **decoder** is a two-layer unidirectional GRU that generates the Bodo translation one token at a time.
- A **Bahdanau attention mechanism** dynamically aligns the decoder with relevant encoder hidden states to improve translation accuracy.

![Model Architecture](image/model_architecture.png)

---

## 🧪 Results

- **Training/Validation Loss**: Shows steady convergence, indicating effective learning.
- **BLEU Scores**:
  - BLEU-1: 14.64
  - BLEU-2: 6.23
  - BLEU-3: 2.89
  - BLEU-4: 1.33

![Training and Validation Loss](image/loss_plot.png)

---

## 🖥️ Interactive Translation Interface

An easy-to-use interface allows users to input English sentences and receive Bodo translations in real-time using the trained model. This facilitates practical usage and testing of the translation system.

![Translation Interface](image/translation_interface.png)

---

## 📁 Dataset

- **Source**: [Alayaran Parallel Corpus](https://get.alayaran.com/parallel-data/)
- **Content**: English–Bodo parallel corpus covering tourism and health domains.
- **Dataset Splits**:
  - Training: 32,149 pairs
  - Validation: 665 pairs
  - Test: 444 pairs

---

## 🛠 Installation & Setup

### 🔧 Prerequisites

- Python 3.8+
- Git
- Jupyter Notebook
- (Optional) GPU with CUDA for faster training

---

### 📦 Python Dependencies

| Package       | Purpose                                |
|---------------|----------------------------------------|
| `torch`       | Neural network implementation           |
| `torchtext`   | Text processing and vocabulary handling |
| `numpy`       | Numerical computing                     |
| `nltk`        | Tokenization, BLEU scoring              |
| `matplotlib`  | Plotting training results               |
| `pickle`      | Saving/loading model and tokenizers     |

---

### ✅ Setup Options

#### 🧪 Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate english-bodo-nmt
jupyter notebook


#### 🧪 Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/akheruddinahmed/English-to-Bodo-Neural-Machine-Translatio.git
cd english-bodo-nmt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install torch torchtext numpy nltk matplotlib

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run Jupyter
jupyter notebook
```

---

## 🚀 How to Use

- **Train model**: Run `model.ipynb`
- **Load & test**: Use `Model Load.ipynb` with `english_processor.pkl` and `bodo_processor.pkl`
- **Loss analysis**: See `loss_plot.png`

---

## 📈 Limitations & Future Work

- Limited training data lowers fluency for long sentences
- BLEU scores indicate word-level success but weak multi-word alignment

### 🔮 Planned Improvements

- Subword tokenization (Byte Pair Encoding)
- Use of pretrained models (e.g., mBART, mT5)
- Back-translation for data augmentation
- Human evaluation metrics
- Web/mobile deployment

---


## 📄 License

This project is for academic and research purposes only.
