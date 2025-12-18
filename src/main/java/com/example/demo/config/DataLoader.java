package com.example.demo.config;

import com.example.demo.entity.Course;
import com.example.demo.entity.Lesson;


import com.example.demo.repository.CourseRepository;
import com.example.demo.repository.LessonRepository;

// NEW imports for AI Tools
import com.example.demo.entity.AiTool;
import com.example.demo.entity.AiToolDetail;
import com.example.demo.repository.AiToolRepository;
import com.example.demo.repository.AIToolDetailRepository;

import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class DataLoader implements CommandLineRunner {

    private final CourseRepository courseRepository;
    private final LessonRepository lessonRepository;

    // NEW repositories for AI Tools
    private final AiToolRepository aiToolRepository;
    private final AIToolDetailRepository aiToolDetailRepository;

    public DataLoader(
            CourseRepository courseRepository,
            LessonRepository lessonRepository,
            AiToolRepository aiToolRepository,
            AIToolDetailRepository aiToolDetailRepository
    ) {
        this.courseRepository = courseRepository;
        this.lessonRepository = lessonRepository;
        this.aiToolRepository = aiToolRepository;
        this.aiToolDetailRepository = aiToolDetailRepository;
    }

    @Override
    public void run(String... args) throws Exception {

        // -----------------------
        // Courses / Lessons / Quiz
        // -----------------------
        boolean hasCourses = courseRepository.count() > 0;
        if (!hasCourses) {
            System.out.println("🚀 Loading Java Full Stack Course Data...");

            // 1️⃣ COURSE
            Course javaCourse = new Course();
            javaCourse.setTitle("Java Full Stack Development");
            javaCourse.setCategory("Programming & Software Development");
            javaCourse.setDescription("Complete Java Full Stack course covering Java, Spring Boot, REST APIs, React, and SQL.");
            courseRepository.save(javaCourse);

            // 2️⃣ LESSONS
            Lesson l1 = new Lesson(
                    "Introduction to Java",
                    "Java is a high-level, object-oriented programming language...",
                    "public class Hello {\n  public static void main(String[] args) {\n    System.out.println(\"Hello Java!\");\n  }\n}"
            );
            l1.setCourse(javaCourse);
            lessonRepository.save(l1);

            Lesson l2 = new Lesson(
                    "Spring Boot REST APIs",
                    "Spring Boot helps build REST APIs quickly using annotations...",
                    "@RestController\n@GetMapping(\"/hello\")\npublic String hello(){ return \"Hello API\"; }"
            );
            l2.setCourse(javaCourse);
            lessonRepository.save(l2);

            Lesson l3 = new Lesson(
                    "React Frontend Basics",
                    "React is a JavaScript library for building UI.",
                    "function App() {\n  return <h1>Hello React</h1>;\n}"
            );
            l3.setCourse(javaCourse);
            lessonRepository.save(l3);

    

            // --- Attach quiz to lesson l1 ---
            // Note: your Quiz entity must have proper mapping to Lesson (quiz.setLesson exists).
            // If your Quiz entity maps to Course instead, adapt this section accordingly.
             
            // ------------------------------------------------
            // QUESTION 1
            // ------------------------------------------------
         
            // ------------------------------------------------
            // QUESTION 2
            // ------------------------------------------------
      

            // 4️⃣ SAVE QUIZ (Cascade saves Questions + Options)

            System.out.println("✅ Java Full Stack Course Data Loaded Successfully!");
        } else {
            System.out.println("📌 Course data already present. Skipping course/lesson/quiz insertion.");
        }

        // -----------------------
        // AI Tools Section
        // -----------------------
        boolean hasAiTools = aiToolRepository.count() > 0;
        if (!hasAiTools) {
            System.out.println("🚀 Loading AI Tools Data...");

            // 1) TensorFlow
            AiTool tf = new AiTool(
                    "TensorFlow",
                    "Google's open-source deep learning framework for neural networks."
            );
            aiToolRepository.save(tf);

            aiToolDetailRepository.save(new AiToolDetail(
                    "Overview",
                    "TensorFlow is an open-source deep learning library developed by Google. "
                            + "It supports building and training neural networks for tasks such as image recognition, "
                            + "object detection, NLP, and time-series forecasting.",
                    tf
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Key Features",
                    "• Eager & graph modes\n• TensorBoard for visualization\n• Distributed training\n• TPU/GPU support\n• TensorFlow Lite for mobile",
                    tf
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "How to Install TensorFlow",
                    "CPU-only:\n\npip install tensorflow\n\nFor GPU (CUDA/cuDNN) follow the official guide: https://www.tensorflow.org/install",
                    tf
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Quick Start Code (Python)",
                    "```python\nimport tensorflow as tf\nfrom tensorflow import keras\n\nmodel = keras.Sequential([\n  keras.layers.Dense(64, activation='relu', input_shape=(784,)),\n  keras.layers.Dense(10, activation='softmax')\n])\n\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n```",
                    tf
            ));

            // 2) PyTorch
            AiTool pt = new AiTool(
                    "PyTorch",
                    "Flexible deep learning framework preferred in research and many production settings."
            );
            aiToolRepository.save(pt);

            aiToolDetailRepository.save(new AiToolDetail(
                    "Overview",
                    "PyTorch is an open-source deep learning framework with dynamic computation graphs. "
                            + "It's popular in research and for rapidly prototyping models (transformers, CNNs, RNNs).",
                    pt
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "How to Install PyTorch",
                    "CPU-only:\n\npip install torch torchvision torchaudio\n\nFor GPU, follow instructions on https://pytorch.org/get-started/locally/",
                    pt
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Quick Start Code (Python)",
                    "```python\nimport torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(784, 10)\n    def forward(self, x):\n        return torch.relu(self.fc(x))\n\nmodel = SimpleNet()\n```",
                    pt
            ));

            // 3) OpenAI API
            AiTool openai = new AiTool(
                    "OpenAI API",
                    "API access to powerful language models (GPT family) for text, chat, embeddings, and more."
            );
            aiToolRepository.save(openai);

            aiToolDetailRepository.save(new AiToolDetail(
                    "Overview",
                    "OpenAI provides hosted language models via an API. It supports text generation, chat completions, embeddings, and other multimodal features.",
                    openai
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "How to Install (Python)",
                    "pip install openai\n\nSet API key:\nexport OPENAI_API_KEY='your_key' (Linux/macOS)\nWindows (PowerShell): $env:OPENAI_API_KEY='your_key'",
                    openai
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Quick Start Code (Python)",
                    "```python\nfrom openai import OpenAI\nclient = OpenAI(api_key='YOUR_KEY')\nresp = client.chat.completions.create(model='gpt-4', messages=[{'role':'user','content':'Hello'}])\nprint(resp.choices[0].message.content)\n```",
                    openai
            ));

            // 4) Scikit-Learn
            AiTool skl = new AiTool(
                    "Scikit-Learn",
                    "Classic machine learning library for regression, classification, clustering and pipelines."
            );
            aiToolRepository.save(skl);

            aiToolDetailRepository.save(new AiToolDetail(
                    "Overview",
                    "Scikit-Learn is a Python library for conventional machine learning algorithms such as linear models, SVMs, tree-based models, and clustering.",
                    skl
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Installation",
                    "pip install scikit-learn\n\nAlso install numpy and scipy for full functionality.",
                    skl
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Quick Start Code (Python)",
                    "```python\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.datasets import load_iris\n\nX, y = load_iris(return_X_y=True)\nclf = RandomForestClassifier()\nclf.fit(X, y)\nprint(clf.predict(X[:5]))\n```",
                    skl
            ));

            // 5) HuggingFace Transformers
            AiTool hf = new AiTool(
                    "HuggingFace Transformers",
                    "Extensive library of pretrained transformer models for NLP, speech, and vision."
            );
            aiToolRepository.save(hf);

            aiToolDetailRepository.save(new AiToolDetail(
                    "Overview",
                    "HuggingFace provides easy access to thousands of pretrained transformer models and high-level pipelines for text, speech, and vision tasks.",
                    hf
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Installation",
                    "pip install transformers\n\nFor PyTorch backend also install torch: pip install torch",
                    hf
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Quick Start Code (Python)",
                    "```python\nfrom transformers import pipeline\nclassifier = pipeline('sentiment-analysis')\nprint(classifier('I love using HuggingFace!'))\n```",
                    hf
            ));

            // 6) Google Vertex AI
            AiTool vertex = new AiTool(
                    "Google Vertex AI",
                    "Managed ML platform by Google Cloud for training, deploying, and monitoring ML models."
            );
            aiToolRepository.save(vertex);

            aiToolDetailRepository.save(new AiToolDetail(
                    "Overview",
                    "Vertex AI integrates Google Cloud services to simplify end-to-end ML workflows: data labeling, training, model registry, deployment, and monitoring.",
                    vertex
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "How to Use (quick)",
                    "Vertex AI is a cloud service—use gcloud CLI and Vertex SDK. Enable Vertex API in GCP project and authenticate with gcloud.",
                    vertex
            ));

            // 7) LangChain
            AiTool lc = new AiTool(
                    "LangChain",
                    "Framework for developing applications with language models using chains, agents, and memory."
            );
            aiToolRepository.save(lc);

            aiToolDetailRepository.save(new AiToolDetail(
                    "Overview",
                    "LangChain provides abstractions for building applications using LLMs: prompt templates, chains, agents, memory, and connectors.",
                    lc
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Installation",
                    "pip install langchain\n\nAlso install an LLM connector (e.g., openai) and a vector DB client (chromadb/faiss).",
                    lc
            ));

            aiToolDetailRepository.save(new AiToolDetail(
                    "Quick Start Code (Python)",
                    "```python\nfrom langchain import OpenAI, LLMChain, PromptTemplate\nprompt = PromptTemplate(input_variables=['topic'], template='Write a short summary about {topic}')\nllm = OpenAI(api_key='YOUR_KEY')\nchain = LLMChain(llm=llm, prompt=prompt)\nprint(chain.run('LangChain'))\n```",
                    lc
            ));

            System.out.println("✅ AI Tools Data Loaded Successfully!");
        } else {
            System.out.println("📌 AI Tools already present. Skipping AI Tools insertion.");
        }
        
        
    }
}
