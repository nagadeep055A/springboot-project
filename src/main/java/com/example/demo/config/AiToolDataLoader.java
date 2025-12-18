package com.example.demo.config;

import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import com.example.demo.entity.AiTool;
import com.example.demo.entity.AiToolDetail;
import com.example.demo.repository.AiToolRepository;
import com.example.demo.repository.AIToolDetailRepository;

/**
 * Full Detailed DataLoader for AI Tools (7 tools).
 * - Inserts 7 tools with many detail sections each (overview, use-cases, installation,
 *   how-it-works, examples, study path, quick code).
 *
 * Paste this file into:
 * src/main/java/com/example/demo/config/AiToolDataLoader.java
 *
 * Requires:
 * - AiTool(String name, String shortDescription) constructor
 * - AiToolDetail(String sectionTitle, String content, AiTool tool) constructor
 * - AiToolRepository and AiToolDetailRepository beans
 */
@Component
public class AiToolDataLoader implements CommandLineRunner {

    private final AiToolRepository toolRepo;
    private final AIToolDetailRepository detailRepo;

    public AiToolDataLoader(AiToolRepository toolRepo, AIToolDetailRepository detailRepo) {
        this.toolRepo = toolRepo;
        this.detailRepo = detailRepo;
    }

    @Override
    public void run(String... args) throws Exception {

        if (toolRepo.count() > 0) {
            System.out.println("📌 AI Tools already loaded. Skipping...");
            return;
        }

        System.out.println("🚀 Loading FULL AI Tools Data (7 tools) ...");

        // -------------------------------------------------
        // 1) TensorFlow
        // -------------------------------------------------
        AiTool tf = new AiTool(
                "TensorFlow",
                "Google's open-source deep learning framework for neural networks."
        );
        toolRepo.save(tf);

        detailRepo.save(new AiToolDetail(
                "Overview",
                "TensorFlow is an open-source deep learning library developed by Google. "
                        + "It supports building and training neural networks for tasks such as image recognition, "
                        + "object detection, NLP, and time-series forecasting.",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "Key Features",
                "• Eager & graph modes\n• TensorBoard for visualization\n• Distributed training\n• TPU/GPU support\n• TensorFlow Lite for mobile",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "Why TensorFlow is Useful",
                "TensorFlow scales from research prototypes to production deployments. It integrates well with "
                        + "cloud services and has tools for monitoring and serving trained models.",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "How it Works (Simple)",
                "You define computational graphs (or use eager mode) where tensors flow through layers. "
                        + "Layers and operations are composed, then a training loop optimizes model parameters via gradient descent.",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "Installation (Windows / macOS / Linux)",
                "CPU-only (pip):\n\npip install tensorflow\n\nGPU (CUDA) - check compatibility first:\n\npip install tensorflow --upgrade\n\nFor specific GPU builds or versions, follow the official installation guide: https://www.tensorflow.org/install",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "Quick Start Code (Python)",
                "```python\nimport tensorflow as tf\nfrom tensorflow import keras\n\nmodel = keras.Sequential([\n  keras.layers.Dense(64, activation='relu', input_shape=(784,)),\n  keras.layers.Dense(10, activation='softmax')\n])\n\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n# model.fit(x_train, y_train, epochs=5)\n```",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "Practical Examples / Projects",
                "• Image classification with CNNs (MNIST, CIFAR10)\n• Object detection with TF Object Detection API\n• Sequence models for text classification\n• Time-series forecasting using LSTMs",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "Best Way to Study",
                "1) Start with Python and NumPy basics.\n2) Learn Keras high-level API for model building.\n3) Follow small projects: MNIST -> CIFAR10 -> simple NLP tasks.\n4) Learn TensorBoard, data pipelines (tf.data), and model deployment (TF Serving or SavedModel).",
                tf
        ));

        detailRepo.save(new AiToolDetail(
                "Common Pitfalls & Tips",
                "• Ensure correct TensorFlow and CUDA/cuDNN versions for GPU.\n• Use tf.data for scalable input pipelines.\n• Normalize inputs and watch for overfitting (use callbacks, early stopping).",
                tf
        ));


        // -------------------------------------------------
        // 2) PyTorch
        // -------------------------------------------------
        AiTool pt = new AiTool(
                "PyTorch",
                "Flexible deep learning framework preferred in research and many production settings."
        );
        toolRepo.save(pt);

        detailRepo.save(new AiToolDetail(
                "Overview",
                "PyTorch is an open-source deep learning framework with dynamic computation graphs. "
                        + "It's popular in research and for rapidly prototyping models (transformers, CNNs, RNNs).",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "Key Features",
                "• Dynamic computation graph (eager execution)\n• TorchScript for production\n• rich ecosystem: torchvision, torchaudio\n• strong community support",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "Why PyTorch is Useful",
                "PyTorch makes model debugging easy thanks to Pythonic control flow. Many state-of-the-art research models are published in PyTorch first.",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "How it Works (Simple)",
                "Define models by subclassing torch.nn.Module. Forward pass produces outputs; backward() computes gradients; optimizers update weights.",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "Installation (Windows / macOS / Linux)",
                "CPU-only:\n\npip install torch torchvision torchaudio\n\nFor CUDA/GPU, use the selector at https://pytorch.org/get-started/locally/ to get the correct command.",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "Quick Start Code (Python)",
                "```python\nimport torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(784, 10)\n    def forward(self, x):\n        return torch.relu(self.fc(x))\n\nmodel = SimpleNet()\n# loss_fn = nn.CrossEntropyLoss()\n# optimizer = torch.optim.Adam(model.parameters())\n```",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "Practical Examples / Projects",
                "• Transformer-based NLP (fine-tune BERT / GPT)\n• Image segmentation & detection\n• Reinforcement Learning experiments",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "Best Way to Study",
                "1) Get comfortable with Python and tensors.\n2) Build small models with torchvision datasets.\n3) Follow research implementations on GitHub.\n4) Learn TorchScript and deployment options when ready.",
                pt
        ));

        detailRepo.save(new AiToolDetail(
                "Tips",
                "• Use GPU for training large models.\n• Use mixed precision (AMP) for faster training when supported.\n• Leverage pretrained models from HuggingFace when applicable.",
                pt
        ));


        // -------------------------------------------------
        // 3) OpenAI API
        // -------------------------------------------------
        AiTool openai = new AiTool(
                "OpenAI API",
                "API access to powerful language models (GPT family) for text, chat, embeddings, and more."
        );
        toolRepo.save(openai);

        detailRepo.save(new AiToolDetail(
                "Overview",
                "OpenAI provides hosted language models via an API. It supports text generation, chat completions, embeddings, and other multimodal features.",
                openai
        ));

        detailRepo.save(new AiToolDetail(
                "Key Features",
                "• Chat / completion endpoints\n• Embeddings for semantic search\n• Moderation tools\n• Fine-tuning & parameter controls",
                openai
        ));

        detailRepo.save(new AiToolDetail(
                "Why OpenAI API is Useful",
                "It provides access to state-of-the-art LLMs without managing infrastructure; ideal for prototypes and production services requiring high-quality NLP.",
                openai
        ));

        detailRepo.save(new AiToolDetail(
                "How it Works (Simple)",
                "You call endpoints with prompts; the model returns generated text or embeddings. Handle tokens, rate limits, and sanitization server-side.",
                openai
        ));

        detailRepo.save(new AiToolDetail(
                "Installation (Python)",
                "Install client:\n\npip install openai\n\nSet your API key as an environment variable:\n\nexport OPENAI_API_KEY='your_key'\n\n(Windows PowerShell: $env:OPENAI_API_KEY='your_key')",
                openai
        ));

        detailRepo.save(new AiToolDetail(
                "Quick Start Code (Python)",
                "```python\nimport openai\nopenai.api_key = 'YOUR_KEY'\n\nresp = openai.Completion.create(\n  engine='text-davinci-003',\n  prompt='Write a haiku about AI',\n  max_tokens=50\n)\nprint(resp.choices[0].text)\n```",
                openai
        ));

        detailRepo.save(new AiToolDetail(
                "Practical Examples / Projects",
                "• Chatbots and virtual assistants\n• Automated summarization\n• Semantic search using embeddings\n• Code generation and pair programming assistants",
                openai
        ));

        detailRepo.save(new AiToolDetail(
                "Best Way to Study",
                "1) Learn prompt engineering basics.\n2) Build small chatbots and iterate.\n3) Experiment with embeddings and vector databases.\n4) Understand costs and latency for production usage.",
                openai
        ));


        // -------------------------------------------------
        // 4) Scikit-Learn
        // -------------------------------------------------
        AiTool skl = new AiTool(
                "Scikit-Learn",
                "Classic machine learning library for regression, classification, clustering and pipelines."
        );
        toolRepo.save(skl);

        detailRepo.save(new AiToolDetail(
                "Overview",
                "Scikit-Learn is a Python library for conventional machine learning algorithms such as linear models, SVMs, tree-based models, and clustering.",
                skl
        ));

        detailRepo.save(new AiToolDetail(
                "Key Features",
                "• Wide range of algorithms\n• Pipelines for reproducible workflows\n• Preprocessing utilities\n• Model selection & validation",
                skl
        ));

        detailRepo.save(new AiToolDetail(
                "Why Scikit-Learn is Useful",
                "Great for structured data problems and rapid prototyping. It's lightweight and easy to integrate into data science workflows.",
                skl
        ));

        detailRepo.save(new AiToolDetail(
                "Installation",
                "Install via pip:\n\npip install scikit-learn\n\nAlso install numpy and scipy for full functionality.",
                skl
        ));

        detailRepo.save(new AiToolDetail(
                "Quick Start Code (Python)",
                "```python\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.datasets import load_iris\n\nX, y = load_iris(return_X_y=True)\nclf = RandomForestClassifier()\nclf.fit(X, y)\nprint(clf.predict(X[:5]))\n```",
                skl
        ));

        detailRepo.save(new AiToolDetail(
                "Practical Examples / Projects",
                "• Customer churn prediction\n• Fraud detection\n• Feature engineering pipelines\n• Baseline models for comparison with deep learning",
                skl
        ));

        detailRepo.save(new AiToolDetail(
                "Best Way to Study",
                "1) Understand statistics and linear algebra basics.\n2) Implement algorithms on toy datasets.\n3) Learn cross-validation and model selection.\n4) Move to pipelines and feature engineering.",
                skl
        ));


        // -------------------------------------------------
        // 5) HuggingFace Transformers
        // -------------------------------------------------
        AiTool hf = new AiTool(
                "HuggingFace Transformers",
                "Extensive library of pretrained transformer models for NLP, speech, and vision."
        );
        toolRepo.save(hf);

        detailRepo.save(new AiToolDetail(
                "Overview",
                "HuggingFace provides easy access to thousands of pretrained transformer models and high-level pipelines for text, speech, and vision tasks.",
                hf
        ));

        detailRepo.save(new AiToolDetail(
                "Key Features",
                "• Transformers library\n• Model Hub with community models\n• Pipelines for quick inference\n• Trainer API for fine-tuning",
                hf
        ));

        detailRepo.save(new AiToolDetail(
                "Why HuggingFace is Useful",
                "It simplifies use of state-of-the-art models and provides a large ecosystem including datasets and tokenizers.",
                hf
        ));

        detailRepo.save(new AiToolDetail(
                "Installation",
                "Install the transformers library:\n\npip install transformers\n\nFor PyTorch backend also install torch:\n\npip install torch",
                hf
        ));

        detailRepo.save(new AiToolDetail(
                "Quick Start Code (Python)",
                "```python\nfrom transformers import pipeline\nclassifier = pipeline('sentiment-analysis')\nprint(classifier('I love using HuggingFace!'))\n```",
                hf
        ));

        detailRepo.save(new AiToolDetail(
                "Practical Examples / Projects",
                "• Fine-tune BERT for classification\n• Use pipeline APIs for summarization and Q&A\n• Deploy LLMs with optimized tokenizers",
                hf
        ));

        detailRepo.save(new AiToolDetail(
                "Best Way to Study",
                "1) Learn tokenization and tokenizer APIs.\n2) Experiment with pretrained models, then fine-tune on a task.\n3) Explore model hub and community notebooks.",
                hf
        ));


        // -------------------------------------------------
        // 6) Google Vertex AI
        // -------------------------------------------------
        AiTool vertex = new AiTool(
                "Google Vertex AI",
                "Managed ML platform by Google Cloud for training, deploying, and monitoring ML models."
        );
        toolRepo.save(vertex);

        detailRepo.save(new AiToolDetail(
                "Overview",
                "Vertex AI integrates Google Cloud services to simplify end-to-end ML workflows: data labeling, training, model registry, deployment, and monitoring.",
                vertex
        ));

        detailRepo.save(new AiToolDetail(
                "Key Features",
                "• Managed training and deployment\n• AutoML options\n• MLOps features (pipelines, model registry)\n• Support for large models and custom containers",
                vertex
        ));

        detailRepo.save(new AiToolDetail(
                "Why Vertex AI is Useful",
                "Makes it easier to move models from experiment to production with built-in monitoring, scaling, and integration into GCP.",
                vertex
        ));

        detailRepo.save(new AiToolDetail(
                "How to Use / Install (quick)",
                "Vertex AI is a cloud service—no local install. Use gcloud CLI and set up a Google Cloud project:\n\n1) gcloud init\n2) gcloud auth login\n3) Enable Vertex API in GCP Console\n4) Use Vertex SDK or REST APIs for training & prediction",
                vertex
        ));

        detailRepo.save(new AiToolDetail(
                "Practical Examples",
                "• AutoML tabular for tabular predictions\n• Custom training with custom containers\n• Deploying models as endpoints with scaling and monitoring",
                vertex
        ));

        detailRepo.save(new AiToolDetail(
                "Best Way to Study",
                "1) Learn GCP basics and IAM.\n2) Try AutoML on sample datasets.\n3) Explore Vertex Pipelines and model monitoring features.",
                vertex
        ));


        // -------------------------------------------------
        // 7) LangChain
        // -------------------------------------------------
        AiTool lc = new AiTool(
                "LangChain",
                "Framework for developing applications with language models using chains, agents, and memory."
        );
        toolRepo.save(lc);

        detailRepo.save(new AiToolDetail(
                "Overview",
                "LangChain provides abstractions for building applications using LLMs: prompt templates, chains, agents, memory, and integrations with external tools and databases.",
                lc
        ));

        detailRepo.save(new AiToolDetail(
                "Key Features",
                "• Chains for composition\n• Agents to call tools / APIs\n• Memory to store context\n• Connectors for vector DBs and tools",
                lc
        ));

        detailRepo.save(new AiToolDetail(
                "Why LangChain is Useful",
                "It speeds up building production LLM applications by combining prompting, tool use, and state management.",
                lc
        ));

        detailRepo.save(new AiToolDetail(
                "Installation",
                "Install LangChain (Python):\n\npip install langchain\n\nAlso install an LLM connector (e.g., openai) and a vector store (e.g., faiss, chromadb).",
                lc
        ));

        detailRepo.save(new AiToolDetail(
                "Quick Start Code (Python)",
                "```python\nfrom langchain import OpenAI, LLMChain, PromptTemplate\n\nprompt = PromptTemplate(input_variables=['topic'], template='Write a short summary about {topic}')\nllm = OpenAI(api_key='YOUR_KEY')\nchain = LLMChain(llm=llm, prompt=prompt)\nprint(chain.run('LangChain'))\n```",
                lc
        ));

        detailRepo.save(new AiToolDetail(
                "Practical Examples / Projects",
                "• Build an agent that answers questions using web search\n• Semantic search over documents using embeddings + vector DB\n• Chatbots with memory and tool usage",
                lc
        ));

        detailRepo.save(new AiToolDetail(
                "Best Way to Study",
                "1) Learn prompt engineering fundamentals.\n2) Build simple chains and progress to agents.\n3) Integrate a vector DB and create retrieval-augmented generation (RAG) apps.",
                lc
        ));


        System.out.println("✅ FULL AI Tools Data Loaded Successfully!");
    }
}
