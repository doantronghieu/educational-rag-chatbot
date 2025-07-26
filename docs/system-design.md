# System Design - RAG Chatbot

## Architecture Overview

```mermaid
graph TB
    %% User Interface Layer
    User[👤 User] --> Frontend["`
        <b>🖥️ Next.js Frontend</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Chat interface with real-time messaging
        • File upload for knowledge base management
        • Settings and configuration panels
        </div>
    `"]
    
    %% API Gateway
    Frontend --> API["`
        <b>🚀 FastAPI Backend</b>
        <div style='text-align:left; white-space:nowrap;'>
        • /chat - Main conversation interface
        • /upload - Knowledge base document upload
        </div>
    `"]
    
    %% Document Processing Pipeline
    PDF[📄 PDF Documents] --> MinIO["`
        <b>☁️ MinIO PDF Storage</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Persistent object storage for original PDF documents
        </div>
    `"]
    MinIO --> Unstructured["`
        <b>📋 Unstructured.IO PDF Parser</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Extract text and structure from PDF documents
        </div>
    `"]
    Unstructured --> LangChain["`
        <b>🔗 LangChain Text Processing</b>
        <div style='text-align:left; white-space:nowrap;'>
        • LLM orchestration
        </div>
    `"]
    
    %% Embedding Pipeline
    LangChain --> OpenAIEmbed["`
        <b>🧠 OpenAI Embedding Model</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Convert text chunks into vector representations
        </div>
    `"]
    OpenAIEmbed --> Qdrant["`
        <b>🔍 Qdrant Vector Database</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Store and search document embeddings
        • Vector similarity semantic search and context retrieval
        </div>
    `"]
    
    %% RAG Pipeline
    API --> QueryProcess[🔄 Query Processing]
    QueryProcess --> Qdrant
    Qdrant --> ContextRetrieval[📚 Context Retrieval]
    ContextRetrieval --> OpenAILLM["`
        <b>🤖 OpenAI LLM Response Generation</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Generate contextual responses using retrieved information
        • Primary school appropriate tone adaptation
        </div>
    `"]
    
    %% Response & Storage
    OpenAILLM --> ResponsePost[✨ Response Post-processing]
    ResponsePost --> Frontend
    
    %% Chat History
    API --> PostgreSQL["`
        <b>🗄️ PostgreSQL Chat History</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Store conversation history and user sessions
        • Query-response pairs
        </div>
    `"]
    
    %% Evaluation
    ResponsePost --> RAGAS["`
        <b>📊 RAGAS Evaluation Framework</b>
        <div style='text-align:left; white-space:nowrap;'>
        • Faithfulness: Response accuracy vs source content
        • Answer Relevancy: Query-response alignment
        • Context Precision & Recall: Retrieval accuracy
        </div>
    `"]
    
    %% Styling
    classDef storage fill:#e1f5fe,stroke:#333,stroke-width:1px
    classDef processing fill:#f3e5f5,stroke:#333,stroke-width:1px
    classDef ai fill:#fff3e0,stroke:#333,stroke-width:1px
    classDef interface fill:#e8f5e8,stroke:#333,stroke-width:1px
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px
    
    class MinIO,PostgreSQL,Qdrant storage
    class Unstructured,LangChain,QueryProcess,ContextRetrieval,ResponsePost processing
    class OpenAIEmbed,OpenAILLM,RAGAS ai
    class User,Frontend,API interface
```


## Data Flow

### 1. Knowledge Base Ingestion
```
PDF Upload → MinIO Storage → Unstructured.IO → LangChain Chunking → 
OpenAI Embeddings → Qdrant Storage
```

### 2. Query Processing
```
User Query → FastAPI → Query Processing → Qdrant Search → 
Context Retrieval → OpenAI LLM → Response Generation → 
PostgreSQL Logging → Frontend Display
```

### 3. Evaluation Loop
```
Generated Response → RAGAS Evaluation → Metrics Storage → 
Performance Monitoring → System Optimization
```
