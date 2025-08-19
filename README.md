# AI Chatbot with PyTorch Neural Networks

A conversational AI chatbot built from scratch using PyTorch neural networks and natural language processing. The chatbot uses intent classification with a custom neural network to understand user queries and provide contextual responses.

## 🤖 Features

- **Neural Network Intent Classification**: Custom 3-layer feedforward network for understanding user intents
- **Natural Language Processing**: Built-in tokenization and lemmatization using NLTK
- **Function Mapping**: Ability to trigger custom functions based on detected intents
- **Model Persistence**: Save and load trained models with their configurations
- **Interactive Chat Interface**: Command-line conversation interface
- **Training Pipeline**: Complete data preprocessing and model training workflow

## 🛠 Tech Stack

- **Deep Learning**: PyTorch, PyTorch Neural Networks
- **NLP**: NLTK (WordNetLemmatizer, word_tokenize)
- **Data Processing**: NumPy for numerical operations
- **Model Architecture**: Feedforward Neural Network with Dropout
- **Storage**: JSON for intents and model dimensions

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/machapraveen/AI-Chatbot-PyTorch.git
cd "AI Chatbot PyTorch"
```

2. Install required dependencies:
```bash
pip install torch nltk numpy
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 🚀 Usage

### Training the Model

Uncomment the training section in `main.py` and run:

```python
# Train a new model
assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
assistant.parse_intents()
assistant.prepare_data()
assistant.train_model(batch_size=8, lr=0.001, epochs=100)
assistant.save_model('chatbot_model.pth', 'dimensions.json')
```

### Running the Chatbot

```bash
python main.py
```

The chatbot will start an interactive session where you can type messages and receive responses.

## 🧠 Model Architecture

The chatbot uses a custom PyTorch neural network:

```python
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
```

**Architecture Details:**
- Input Layer: Bag-of-words representation
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 64 neurons with ReLU activation
- Output Layer: Number of intent classes
- Regularization: 50% Dropout between layers

## 📊 Intent Configuration

The chatbot understands various intents defined in `intents.json`:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "How are you", "Hello", "Good day"],
      "responses": ["Hello!", "Good to see you again!"]
    },
    {
      "tag": "stocks",
      "patterns": ["What are my stocks?", "Show my stock portfolio"],
      "responses": ["Here are your stocks!"]
    }
  ]
}
```

## 🔧 Core Components

### ChatbotAssistant Class

The main class that handles:
- **Intent Parsing**: Processes intents.json and extracts patterns
- **Data Preprocessing**: Tokenization, lemmatization, and bag-of-words conversion
- **Model Training**: Neural network training with PyTorch
- **Inference**: Message processing and response generation

### Key Methods

```python
# Process user messages
def process_message(self, input_message):
    words = self.tokenize_and_lemmatize(input_message)
    bag = self.bag_of_words(words)
    # Neural network prediction
    predictions = self.model(bag_tensor)
    predicted_intent = self.intents[predicted_class_index]
    return response

# Custom function integration
function_mappings = {'stocks': get_stocks}
```

## 🎯 Training Process

1. **Data Preparation**: Extract patterns from intents and create training data
2. **Tokenization**: Break down sentences into words using NLTK
3. **Lemmatization**: Reduce words to their root form
4. **Bag-of-Words**: Convert text to numerical vectors
5. **Neural Network Training**: Train using Adam optimizer and CrossEntropyLoss

## 📈 Performance Features

- **Dropout Regularization**: Prevents overfitting during training
- **Adam Optimizer**: Efficient gradient-based optimization
- **Batch Processing**: Configurable batch size for training
- **Learning Rate Control**: Adjustable learning rate parameter

## 💡 Example Interactions

```
Enter your message: Hello
> Hello!

Enter your message: What are my stocks?
> Here are your stocks!
['NVDA', 'GS', 'MSFT']  # Custom function output

Enter your message: Tell me about programming
> Programming, coding or software development, means writing computer code to automate tasks.
```

## 🔮 Future Enhancements

- Web interface integration
- More sophisticated NLP preprocessing
- Context-aware conversations
- Database integration for dynamic responses
- Multi-language support

## 👨‍💻 Author

**Macha Praveen**
- GitHub: [@machapraveen](https://github.com/machapraveen)

## 📄 License

This project is open source and available under the MIT License.
