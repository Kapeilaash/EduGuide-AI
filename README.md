# 🎓 EduGuide-AI

> An intelligent AI assistant for university helpdesk queries, fine-tuned specifically for Sri Lankan universities

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-green.svg)](https://huggingface.co/transformers/)

---

## 📋 Table of Contents

- [What is EduGuide-AI?](#-what-is-eduguide-ai)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Installation](#-installation)
- [How to Use](#-how-to-use)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Results & Evaluation](#-results--evaluation)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [References](#-references)

---

## 🤔 What is EduGuide-AI?

EduGuide-AI is an AI-powered assistant that helps students get quick answers to common university-related questions. Whether you need help with hostel applications, LMS password recovery, exam results, or course registration, this assistant is here to help!

**Key Highlights:**
- ✅ Trained on 600+ real university helpdesk queries
- ✅ Optimized for Sri Lankan university systems
- ✅ Fast and efficient (uses only 1-2% of model parameters)
- ✅ Works on free Google Colab GPUs
- ✅ Easy to use and deploy

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Beginners)

1. **Open the notebook** in Google Colab
   - Upload `complete_training_and_inference.ipynb` to Colab
   - Upload `education_university_helpdesk_srilanka_dataset.csv` to the same folder

2. **Run all cells** - The notebook will automatically:
   - Install all required packages
   - Load and prepare the data
   - Train the model
   - Show you the results

3. **Ask questions!** - Use the inference cells to chat with your trained assistant

### Option 2: Local Setup

```bash
# 1. Clone or download this repository
# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook complete_training_and_inference.ipynb
```

---

## ✨ Features

### What Can EduGuide-AI Help With?

- 🏠 **Hostel Applications** - How to apply for university accommodation
- 🔐 **LMS Support** - Password recovery and login issues
- 📊 **Exam Results** - Where and how to check your grades
- 📝 **Course Registration** - Semester course enrollment
- 📅 **Academic Calendar** - Important dates and deadlines
- 💼 **Internships** - Application procedures
- 📚 **Library Services** - Access resources and check hours
- 🏥 **Medical Certificates** - Requirements for exams
- ⏰ **Timetable Issues** - Class schedule problems
- 📖 **General Queries** - Any university-related question

### Why Choose This Approach?

- **Efficient Training**: Uses LoRA/PEFT to train only 1-2% of parameters
- **Fast Inference**: Quick responses in seconds
- **Resource-Friendly**: Works on free Colab GPUs
- **Domain-Specific**: Trained specifically for university helpdesk context

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Google Colab account (recommended) OR local environment with GPU
- Basic knowledge of Python (helpful but not required)

### Step-by-Step Installation

#### For Google Colab Users:

The notebook will automatically install everything you need! Just run the first cell.

#### For Local Users:

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install transformers datasets peft accelerate bitsandbytes torch pandas matplotlib seaborn scikit-learn
```

### Required Files

Make sure you have these files in your project folder:

```
📁 Your Project Folder
├── 📄 complete_training_and_inference.ipynb    # Main notebook
├── 📄 education_university_helpdesk_srilanka_dataset.csv  # Training data
├── 📄 requirements.txt                         # Dependencies
└── 📄 README.md                                # This file
```

---

## 🎯 How to Use

### Training the Model

1. **Open the notebook** (`complete_training_and_inference.ipynb`)

2. **Run cells in order** - The notebook is divided into clear sections:
   - 📊 Data Preparation
   - ⚙️ Model Setup
   - 🏋️ Training
   - 📈 Evaluation
   - 💬 Inference

3. **Wait for training** - This typically takes 10-30 minutes depending on your hardware

4. **Check results** - View the training loss curve and model comparisons

### Using the Trained Model

#### Simple Question-Answer

```python
# Load the trained model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./flan-t5-university-helpdesk-lora-final")
model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-university-helpdesk-lora-final")

# Ask a question
question = "How do I apply for hostel facilities?"
formatted_input = f"answer: {question}"

# Get response
inputs = tokenizer(formatted_input, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Question: {question}")
print(f"Answer: {response}")
```

#### Interactive Mode

The notebook includes an interactive mode where you can chat with the assistant:

```python
# Uncomment and run this in the notebook
interactive_mode(fine_tuned_model, tokenizer, device)
```

---

## 📁 Project Structure

```
EduGuide-AI/
│
├── 📄 complete_training_and_inference.ipynb    # Main training & inference notebook
├── 📄 education_university_helpdesk_srilanka_dataset.csv  # Training dataset (600 samples)
├── 📄 requirements.txt                         # Python dependencies
├── 📄 README.md                                # Project documentation
│
├── 📁 flan-t5-university-helpdesk-lora-final/  # Saved model (after training)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
│
├── 📁 logs/                                     # Training logs (after training)
└── 📄 training_loss_curve.png                  # Training visualization (after training)
```

---

## 🔧 Technical Details

### Model Information

| Component | Details |
|-----------|---------|
| **Base Model** | `google/flan-t5-base` (250M parameters) |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) / PEFT |
| **Trainable Parameters** | ~1.7M (only 0.7% of total) |
| **Dataset Size** | 600 instruction-response pairs |
| **Training Time** | 10-30 minutes (depending on hardware) |

### Why FLAN-T5-base?

- ✅ Already instruction-tuned (perfect for Q&A tasks)
- ✅ Small enough to train quickly
- ✅ Works great on free Colab GPUs
- ✅ Proven performance on similar tasks

### Why LoRA/PEFT?

- ⚡ **Faster Training** - Only trains 1-2% of parameters
- 💾 **Less Memory** - Uses less GPU memory
- 🎯 **Prevents Overfitting** - Fewer parameters = less overfitting
- 🔄 **Flexible** - Easy to adapt to new tasks

### Training Configuration

```
Epochs: 3
Batch Size: 8 per device
Learning Rate: 0.0005
Max Input Length: 512 tokens
Max Output Length: 128 tokens
```

---

## 📊 Results & Evaluation

### What We Tested

The model was evaluated on 12+ diverse test questions covering:
- Hostel applications
- LMS password recovery
- Exam results
- Course registration
- Academic calendar
- And more...

### Improvements Observed

✅ **Better Context Understanding** - Understands university-specific terms (LMS, UGC, Student Affairs, etc.)  
✅ **More Relevant Answers** - Responses are tailored to Sri Lankan university systems  
✅ **Consistent Format** - Uniform tone and structure in responses  
✅ **Domain Knowledge** - Better understanding of helpdesk context

### Current Limitations

⚠️ **Model Size** - 250M parameters may sometimes give generic responses  
⚠️ **Dataset Size** - 600 samples may not cover all possible queries  
⚠️ **Complex Questions** - May struggle with multi-part questions  
⚠️ **Real-time Info** - Cannot provide current information (e.g., today's library hours)

### Sample Outputs

**Question:** "How do I apply for hostel facilities?"

**Answer:** "Hostel applications are usually handled by the Student Affairs Division. Notices are published before each academic year."

---

## ❓ FAQ

### Q: Do I need a GPU to train this model?

**A:** Not required, but highly recommended! Training on CPU will work but will be much slower (could take hours). Google Colab provides free GPU access.

### Q: How long does training take?

**A:** 
- On GPU (Colab): 10-30 minutes
- On CPU: 1-3 hours (depending on your computer)

### Q: Can I use this for other universities?

**A:** Yes! The model is trained on Sri Lankan university data, but you can retrain it with your own dataset for any university or domain.

### Q: What if I want better results?

**A:** Try these improvements:
- Use a larger model (FLAN-T5-large)
- Add more training data (1000+ samples)
- Train for more epochs
- Fine-tune hyperparameters

### Q: Can I deploy this as a chatbot?

**A:** Yes! The model can be integrated into web applications, chatbots, or APIs. Check the inference section for code examples.

### Q: Is the dataset included?

**A:** Yes! The dataset file `education_university_helpdesk_srilanka_dataset.csv` is included in the repository.

---

## 🤝 Contributing

This project was created for the OXZON AI fine-tuning task evaluation. 

**Want to improve it?** Here are some ideas:
- Add more training data
- Improve the model architecture
- Add evaluation metrics (BLEU, ROUGE)
- Create a web interface
- Add support for multiple languages

---

## 📚 References

### Papers & Documentation

- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416) - Original FLAN-T5 research
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation technique
- [PEFT Library](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model library

### Useful Resources

- [HuggingFace Model Hub](https://huggingface.co/models)
- [Google Colab](https://colab.research.google.com/) - Free GPU access
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

Made with ❤️ for the education community

</div>
