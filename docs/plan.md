# RAG Chatbot Assessment - Project Plan

## Gantt Chart

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize': '30px'}}}%%
gantt
    title RAG Chatbot Assessment Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %m/%d
    
    section Day 1
    Environment Config  :done, env, 2025-07-27, 1d
    Dependencies Install  :done, deps, 2025-07-27, 1d
    Project Structure    :done, struct, 2025-07-27, 1d
    
    section Day 2
    System Architecture :arch, 2025-07-28, 1d
    Database Vector Store :db, 2025-07-28, 1d
    API Structure Plan     :api, 2025-07-28, 1d
    
    section Day 3
    PDF KB Loading :kb, 2025-07-29, 1d
    Embedding Retrieval :rag, 2025-07-29, 1d
    LLM Integration           :llm, 2025-07-29, 1d
    Frontend UI Dev   :ui, 2025-07-29, 1d
    
    section Day 4
    Tone Adaptation :tone, 2025-07-30, 1d
    RAGAS Framework :ragas, 2025-07-30, 1d
    Performance Opt  :optimize, 2025-07-30, 1d
    Deploy and Docs :deploy, 2025-07-30, 1d
```

## Project Overview

This Gantt chart outlines the development timeline for the RAG Chatbot Assessment, spanning 4 intensive days from July 27-30, 2025.

### Key Phases:

1. **Day 1: Project Setup (July 27)**
   - Environment configuration and dependencies
   - Project structure scaffolding
   - Initial setup completion

2. **Day 2: Architecture Design and Setup (July 28)**
   - System architecture design
   - Database and vector store configuration
   - API structure planning

3. **Day 3: RAG Implementation and UI (July 29)**
   - PDF knowledge base processing
   - Embedding and retrieval system implementation
   - LLM integration and frontend UI development

4. **Day 4: Improvement and Deployment (July 30)**
   - Primary school tone adaptation
   - RAGAS evaluation framework implementation
   - Performance optimization and deployment

### Success Criteria:
- RAGAS scores >80% on faithfulness, answer relevancy, context precision, and context recall
- Primary school appropriate tone and language
- Proper scope limitation with fallback responses
- Complete documentation and working deployment