"""
COMPREHENSIVE EXAMPLES COVERING ALL LANGCHAIN RETRIEVER FUNCTIONALITY

Importing shared document collections from test data
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))

# Import shared document collections
from backend.tests.data.retriever_test_docs import (
    get_programming_docs, get_ml_ai_docs, get_cloud_computing_docs, 
    get_python_overlap_docs, get_apple_docs, get_task_decomposition_docs,
    get_celtics_docs, get_movie_docs, get_travel_docs, get_compression_docs,
    get_ai_filter_docs, get_long_ml_docs, get_computing_history_docs
)

from backend.libs.langchain.retrievers import Retriever, RetrieverManager
from backend.core.dependencies import get_embeddings, get_llm

# Common test setup - shared across all functions to reduce redundancy
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# Setup shared components
embeddings = get_embeddings()
vectorstore = InMemoryVectorStore(embeddings)
llm = get_llm().model

"""
1. VECTOR STORE RETRIEVERS
==========================
"""

# Basic similarity search
def example_similarity_retriever():
    """Example: Basic similarity search retriever"""
    # Setup vector store with diverse, realistic content
    local_vectorstore = InMemoryVectorStore(embeddings)
    docs = get_programming_docs()
    local_vectorstore.add_documents(docs)
    
    # Create similarity retriever
    retriever = Retriever.create_vector_retriever(
        vectorstore=local_vectorstore,
        search_type="similarity",
        k=3  # Get top 3 most similar documents
    )
    
    # Query for programming-related content - should return Python, JavaScript, and possibly web dev docs
    results = retriever.invoke("programming languages for beginners")
    print(f"Found {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:60]}...")
    return results

# MMR (Maximum Marginal Relevance) search
def example_mmr_retriever():
    """Example: MMR retriever for diverse results - reduces redundancy"""
    # Use documents with intentional overlap to demonstrate MMR's diversity benefit
    local_vectorstore = InMemoryVectorStore(embeddings)
    docs = get_python_overlap_docs()
    local_vectorstore.add_documents(docs)
    
    # Create MMR retriever - will select diverse documents, avoiding similar Python docs
    retriever = Retriever.create_vector_retriever(
        vectorstore=local_vectorstore,
        search_type="mmr",
        k=4,
        fetch_k=8,  # Fetch more docs then filter for diversity
        lambda_mult=0.5  # Balance between relevance and diversity
    )
    
    # Query about programming - MMR should return diverse languages, not just Python variants
    results = retriever.invoke("programming languages and their features")
    print(f"MMR returned {len(results)} diverse documents:")
    for i, doc in enumerate(results, 1):
        language = doc.page_content.split()[0]  # First word is usually the language
        print(f"{i}. {language}: {doc.page_content[:50]}...")
    return results

# Score-based retrieval
def example_score_retriever():
    """Example: Retriever that includes similarity scores for ranking validation"""
    # Use ML/AI documents with varying relevance to test query
    local_vectorstore = InMemoryVectorStore(embeddings)
    docs = get_ml_ai_docs()
    local_vectorstore.add_documents(docs)
    
    # Create score retriever
    retriever = Retriever.create_score_retriever(
        vectorstore=local_vectorstore,
        score_key="similarity_score",
        k=4
    )
    
    # Query specifically about machine learning - should show clear score differences
    results = retriever.invoke("machine learning and artificial intelligence")
    print(f"Retrieved {len(results)} documents with similarity scores:")
    for i, doc in enumerate(results, 1):
        score = doc.metadata.get("similarity_score", "N/A")
        print(f"{i}. Score: {score:.3f} - {doc.page_content[:60]}...")
    
    # Results will have scores in metadata: doc.metadata["similarity_score"]
    # Higher scores = more similar to query
    return results

# Similarity score threshold
def example_threshold_retriever():
    """Example: Filter results by similarity threshold - only high-confidence matches"""
    # Use cloud computing documents with mixed relevance levels
    local_vectorstore = InMemoryVectorStore(embeddings)
    docs = get_cloud_computing_docs()
    local_vectorstore.add_documents(docs)
    
    # Compare regular similarity search vs threshold-based filtering
    print("Regular similarity search (no threshold):")
    regular_retriever = Retriever.create_vector_retriever(
        vectorstore=local_vectorstore,
        search_type="similarity",
        k=5
    )
    regular_results = regular_retriever.invoke("cloud computing services")
    for i, doc in enumerate(regular_results, 1):
        relevance = "☁️ Cloud" if "cloud" in doc.page_content.lower() else "❌ Other"
        print(f"{i}. {relevance}: {doc.page_content[:50]}...")
    
    # Create threshold-based retriever - only high similarity scores
    print("\nWith similarity threshold (score_threshold=0.7):")
    retriever = Retriever.create_vector_retriever(
        vectorstore=local_vectorstore,
        search_type="similarity",
        score_threshold=0.7  # Only return docs with similarity >= 0.7
    )
    
    results = retriever.invoke("cloud computing services")
    print(f"Filtered to {len(results)} high-confidence matches:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. ☁️ High confidence: {doc.page_content[:60]}...")
    
    return results

"""
2. MULTI-QUERY RETRIEVER
=========================
"""

def example_multi_query_retriever():
    """Example: Multi-query retriever with LLM-generated query variations"""
    # Setup components
    local_vectorstore = InMemoryVectorStore(embeddings)
    
    # Use task decomposition documents (like in LangChain docs example)
    docs = get_task_decomposition_docs()
    local_vectorstore.add_documents(docs)
    
    # Create base retriever
    base_retriever = local_vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create multi-query retriever
    retriever = Retriever.create_multi_query_retriever(
        base_retriever=base_retriever,
        llm=llm
    )
    
    # This will generate multiple query variations like:
    # - "How can Task Decomposition be achieved through different methods?"
    # - "What strategies are used for breaking down complex tasks?"
    # - "What are various approaches to decompose complex problems?"
    print("Multi-query retriever will generate variations of the query...")
    results = retriever.invoke("What are the approaches to Task Decomposition?")
    
    print(f"Retrieved {len(results)} unique documents from multiple query variations:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:70]}...")
    
    return results

"""
3. CONTEXTUAL COMPRESSION
==========================
"""

def example_llm_compression_retriever():
    """Example: LLM-based contextual compression - extracts relevant content"""
    # Setup components
    local_vectorstore = InMemoryVectorStore(embeddings)
    
    # Create documents with lots of content, only some relevant to queries
    docs = [
        Document(page_content="""Judge Ketanji Brown Jackson was nominated to the Supreme Court by President Biden in February 2022. She previously served on the U.S. Court of Appeals for the D.C. Circuit and the U.S. District Court for the District of Columbia. Before her judicial career, Jackson worked as a federal public defender and was a partner at a law firm. She graduated from Harvard Law School where she served as an editor of the Harvard Law Review. Jackson is known for her thorough preparation and thoughtful questioning during hearings. The nomination process involved extensive Senate hearings where she answered questions about her judicial philosophy, past decisions, and approach to constitutional interpretation. Her confirmation made her the first Black woman to serve on the Supreme Court."""),
        
        Document(page_content="""The Supreme Court of the United States consists of nine justices who serve life tenure. The Court's primary function is to interpret the Constitution and federal law. Cases reach the Court through a writ of certiorari, where the justices vote on which cases to hear. The Court typically hears 60-80 cases per year out of thousands of petitions. Justices write majority opinions, concurring opinions, and dissenting opinions to explain their reasoning. The Court's decisions are binding on all lower courts and establish legal precedent. Famous landmark cases include Brown v. Board of Education, Roe v. Wade, and Miranda v. Arizona. The building where the Court sits was completed in 1935 and features neoclassical architecture. Court sessions run from October through June, with summer recess for writing opinions.""")
    ]
    local_vectorstore.add_documents(docs)
    
    # Create base retriever
    base_retriever = local_vectorstore.as_retriever()
    
    # Create LLM compression retriever
    retriever = Retriever.create_compression_retriever(
        base_retriever=base_retriever,
        compressor_type="llm",
        llm=llm
    )
    
    print("Before compression - full documents retrieved:")
    base_results = base_retriever.invoke("Ketanji Brown Jackson Supreme Court nomination")
    for i, doc in enumerate(base_results, 1):
        word_count = len(doc.page_content.split())
        print(f"{i}. Full document ({word_count} words): {doc.page_content[:80]}...")
    
    print("\nAfter LLM compression - only relevant content extracted:")
    # This will compress documents to only the parts relevant to the query
    results = retriever.invoke("Ketanji Brown Jackson Supreme Court nomination")
    
    for i, doc in enumerate(results, 1):
        word_count = len(doc.page_content.split())
        print(f"{i}. Compressed ({word_count} words): {doc.page_content}")
    
    return results

def example_embeddings_compression_retriever():
    """Example: Embeddings-based compression filter - filters by similarity threshold"""
    # Setup components
    local_vectorstore = InMemoryVectorStore(embeddings)
    
    # Create documents with varying relevance to test query
    docs = [
        Document(page_content="Artificial intelligence and machine learning are transforming modern technology and business applications."),
        Document(page_content="Deep learning neural networks use multiple layers to process complex patterns in data and images."),
        Document(page_content="Natural language processing enables computers to understand and generate human language effectively."),
        Document(page_content="The weather today is sunny and perfect for outdoor activities like hiking and picnics."),
        Document(page_content="Cooking delicious pasta requires proper timing, quality ingredients, and attention to detail."),
        Document(page_content="Computer vision algorithms can identify objects, faces, and scenes in digital images and videos."),
        Document(page_content="Music streaming services have changed how people discover and listen to their favorite songs."),
        Document(page_content="Data science combines statistics, programming, and domain knowledge to extract insights from data.")
    ]
    local_vectorstore.add_documents(docs)
    
    # Create base retriever that gets more documents
    base_retriever = local_vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Create embeddings compression retriever with moderate similarity threshold
    retriever = Retriever.create_compression_retriever(
        base_retriever=base_retriever,
        compressor_type="embeddings",
        embeddings=embeddings,
        similarity_threshold=0.5,  # Moderate threshold - should return relevant docs
        k=3  # Maximum 3 documents after filtering
    )
    
    print("Before embeddings filtering - base retrieval:")
    base_results = base_retriever.invoke("machine learning and AI technology")
    for i, doc in enumerate(base_results, 1):
        relevance = "🤖 AI/ML" if any(term in doc.page_content.lower() for term in ["ai", "machine", "learning", "neural", "data"]) else "❌ Other"
        print(f"{i}. {relevance}: {doc.page_content[:60]}...")
    
    print(f"\nAfter embeddings filtering (threshold=0.5) - only highly similar docs:")
    # This will filter documents based on embedding similarity threshold
    results = retriever.invoke("machine learning and AI technology")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. 🎯 Highly relevant: {doc.page_content[:70]}...")
    
    print(f"\nFiltered from {len(base_results)} to {len(results)} documents based on similarity threshold.")
    
    return results

"""
4. ENSEMBLE RETRIEVER
======================
"""

def example_ensemble_retriever():
    """Example: Ensemble retriever combining semantic and keyword search"""
    from langchain_community.retrievers import BM25Retriever
    
    # Setup documents with content that shows difference between keyword and semantic search
    docs = [
        Document(page_content="Fresh apples are delicious fruits rich in vitamins and fiber. Red apples and green apples both offer great nutritional value."),
        Document(page_content="Apple Inc. is a technology company known for innovative products like the iPhone, iPad, and Mac computers."),
        Document(page_content="Fruit orchards grow many varieties of apples including Granny Smith, Red Delicious, and Honeycrisp apples."),
        Document(page_content="The apple tree is a deciduous tree in the rose family best known for its sweet, pomaceous fruit, the apple."),
        Document(page_content="Steve Jobs co-founded Apple Computer Inc. and helped transform it into the world's most valuable technology company."),
        Document(page_content="Nutritionists recommend eating apples daily as they contain antioxidants and promote digestive health."),
        Document(page_content="Technology stocks like Apple, Google, and Microsoft have driven market growth in recent years."),
        Document(page_content="Orchard management involves pruning apple trees, pest control, and harvesting at optimal ripeness.")
    ]
    
    # Create vector retriever (semantic similarity)
    local_vectorstore = InMemoryVectorStore(embeddings)
    local_vectorstore.add_documents(docs)
    vector_retriever = local_vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Create BM25 retriever (keyword matching)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4
    
    # Create ensemble retriever that combines both approaches
    retriever = Retriever.create_ensemble_retriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],  # Weight semantic search more heavily
        c=60  # Reciprocal rank fusion parameter
    )
    
    # Query that benefits from both keyword and semantic matching
    # Keyword search finds "apples", semantic search finds related fruit content
    print("Ensemble retriever combines semantic and keyword search...")
    results = retriever.invoke("healthy apples nutrition")
    
    print(f"Ensemble retrieved {len(results)} documents combining both search methods:")
    for i, doc in enumerate(results, 1):
        doc_type = "🍎 Fruit" if "fruit" in doc.page_content.lower() or "nutrition" in doc.page_content.lower() else "💻 Tech"
        print(f"{i}. {doc_type}: {doc.page_content[:60]}...")
    
    return results

"""
5. LONG CONTEXT REORDER
========================
"""

def example_reorder_retriever():
    """Example: Long context reorder to mitigate 'lost in the middle' effect"""
    # Setup documents about Boston Celtics (like in LangChain docs example) with varying relevance
    local_vectorstore = InMemoryVectorStore(embeddings)
    docs = [
        Document(page_content="The Boston Celtics are my favorite NBA team and have won 17 championships."),
        Document(page_content="This is a document about the Boston Celtics basketball team and their history."),
        Document(page_content="The Boston Celtics won the game by 20 points in a dominant performance."),
        Document(page_content="I simply love going to the movies on weekends with friends and family."),
        Document(page_content="Weather today is sunny and perfect for outdoor activities like hiking."),
        Document(page_content="Cooking pasta is easy when you follow the right recipe and timing."),
        Document(page_content="The Celtics have legendary players like Larry Bird, Bill Russell, and Paul Pierce."),
        Document(page_content="Basketball is a popular sport played worldwide with professional leagues."),
        Document(page_content="Green is a color that represents nature, growth, and the Boston Celtics."),
        Document(page_content="TD Garden is the home arena of the Boston Celtics basketball team.")
    ]
    local_vectorstore.add_documents(docs)
    
    # Create base retriever that returns many documents
    base_retriever = local_vectorstore.as_retriever(search_kwargs={"k": 8})
    
    # Create reorder retriever
    retriever = Retriever.create_reorder_retriever(
        base_retriever=base_retriever
    )
    
    print("Before reordering: relevant docs might be buried in middle...")
    base_results = base_retriever.invoke("Boston Celtics basketball team")
    for i, doc in enumerate(base_results, 1):
        relevance = "🏀 Relevant" if "celtics" in doc.page_content.lower() else "❌ Irrelevant"
        print(f"{i}. {relevance}: {doc.page_content[:50]}...")
    
    # This will reorder documents: most relevant at beginning/end, less relevant in middle
    print("\nAfter reordering: relevant docs moved to beginning and end...")
    results = retriever.invoke("Boston Celtics basketball team")
    for i, doc in enumerate(results, 1):
        relevance = "🏀 Relevant" if "celtics" in doc.page_content.lower() else "❌ Irrelevant"
        print(f"{i}. {relevance}: {doc.page_content[:50]}...")
    
    return results

"""
6. SELF-QUERY RETRIEVER
========================
"""

async def example_self_query_retriever():
    """Example: Self-query retriever with metadata filtering using QdrantVectorStore"""
    from langchain.chains.query_constructor.base import AttributeInfo
    from backend.core.dependencies import get_vector_store_service
    
    try:
        # Get vector store service which provides QdrantVectorStore (supports self-query)
        vector_service = get_vector_store_service()
        
        # Setup documents with metadata for movie database
        docs = [
            Document(
                page_content="Interstellar is a science fiction epic about space exploration and time dilation, directed by Christopher Nolan.",
                metadata={"genre": "sci-fi", "year": 2014, "rating": 8.6, "director": "Christopher Nolan"}
            ),
            Document(
                page_content="The Grand Budapest Hotel is a whimsical comedy-drama about a legendary concierge and his protégé.",
                metadata={"genre": "comedy", "year": 2014, "rating": 8.1, "director": "Wes Anderson"}
            ),
            Document(
                page_content="Blade Runner 2049 is a neo-noir science fiction film set in a dystopian future.",
                metadata={"genre": "sci-fi", "year": 2017, "rating": 8.0, "director": "Denis Villeneuve"}
            ),
            Document(
                page_content="The Princess Bride is a romantic adventure comedy that has become a cult classic.",
                metadata={"genre": "comedy", "year": 1987, "rating": 8.0, "director": "Rob Reiner"}
            ),
            Document(
                page_content="Dune is an epic science fiction film about political intrigue on a desert planet.",
                metadata={"genre": "sci-fi", "year": 2021, "rating": 8.0, "director": "Denis Villeneuve"}
            )
        ]
        
        # Add documents to vector store asynchronously  
        import time
        collection_name = f"movies_test_{int(time.time())}"  # Unique collection name
        
        # Add documents and get vector store
        await vector_service.add_documents(docs, collection_name=collection_name)
        vectorstore = await vector_service.get_vector_store(collection_name)
        
        # Define metadata fields for self-query
        metadata_field_info = [
            AttributeInfo(
                name="genre",
                description="The genre of the movie (sci-fi, comedy, etc.)",
                type="string"
            ),
            AttributeInfo(
                name="year", 
                description="The year the movie was released",
                type="integer"
            ),
            AttributeInfo(
                name="rating",
                description="Movie rating from 1-10 scale",
                type="float"
            ),
            AttributeInfo(
                name="director",
                description="The director of the movie",
                type="string"
            )
        ]
        
        # Create self-query retriever with QdrantVectorStore
        retriever = Retriever.create_self_query_retriever(
            vectorstore=vectorstore,
            document_content_description="Movie descriptions with genre, year, rating, and director information",
            metadata_field_info=metadata_field_info,
            llm=llm
        )
        
        print("Self-query retriever using QdrantVectorStore...")
        print("Testing natural language query with metadata filtering...")
        
        # Test various self-query examples
        queries = [
            "sci-fi movies with high ratings",
            "comedy movies from before 2000", 
            "movies directed by Denis Villeneuve",
            "movies with rating above 8.5"
        ]
        
        all_results = []
        for query in queries:
            print(f"\nQuery: '{query}'")
            try:
                results = retriever.invoke(query)
                print(f"Found {len(results)} matching movies:")
                for i, doc in enumerate(results, 1):
                    metadata = doc.metadata
                    print(f"{i}. {metadata.get('genre', 'N/A')} ({metadata.get('year', 'N/A')}) - Rating: {metadata.get('rating', 'N/A')}")
                    print(f"   {doc.page_content[:80]}...")
                all_results.extend(results)
            except Exception as e:
                print(f"Query failed: {e}")
        
        return all_results
        
    except Exception as e:
        print(f"Self-query test failed (this may be expected if Qdrant is not available): {e}")
        print("Falling back to regular similarity search...")
        
        # Fallback to regular similarity search
        from langchain_core.vectorstores import InMemoryVectorStore
        local_vectorstore = InMemoryVectorStore(embeddings)
        docs = [
            Document(page_content="Science fiction movie about space exploration"),
            Document(page_content="Romantic comedy from the 1990s")
        ]
        local_vectorstore.add_documents(docs)
        
        fallback_retriever = Retriever.create_vector_retriever(
            vectorstore=local_vectorstore,
            search_type="similarity",
            k=2
        )
        
        results = fallback_retriever.invoke("sci-fi movies with high ratings")
        print(f"Fallback similarity search returned {len(results)} documents")
        return results
    
    finally:
        # Clean up test collection if it was created
        try:
            if 'collection_name' in locals() and 'vector_service' in locals():
                # Delete the test collection to clean up
                vector_client = vector_service.client
                if await vector_client.collection_exists(collection_name):
                    await vector_client.delete_collection(collection_name)
                    print(f"Cleaned up test collection: {collection_name}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up test collection: {cleanup_error}")

"""
7. MULTI-VECTOR RETRIEVER
==========================
"""

def example_multi_vector_retriever():
    """Example: Multi-vector retriever - index summaries/chunks, return full docs"""
    from langchain.storage import InMemoryStore
    import uuid
    
    # Setup components
    local_vectorstore = InMemoryVectorStore(embeddings)
    docstore = InMemoryStore()
    
    # Create multi-vector retriever
    retriever = Retriever.create_multi_vector_retriever(
        vectorstore=local_vectorstore,
        docstore=docstore,
        id_key="doc_id"
    )
    
    # Create full documents (what gets returned)
    full_docs = [
        Document(page_content="""The IBM 1401 Computer System: In the early days of computing, the IBM 1401 was a variable word length decimal computer that was announced by IBM on October 5, 1959. The system was widely used for business applications and was known for its reliability and ease of programming. Programming the IBM 1401 required understanding its unique instruction set and memory organization. The computer used a magnetic core memory system and supported various input/output devices including card readers, line printers, and magnetic tape drives. Many programmers got their start on systems like the IBM 1401, learning fundamental concepts that would serve them throughout their careers in computing."""),
        
        Document(page_content="""Modern Programming Languages: Today's programming landscape is dominated by languages like Python, JavaScript, and Java. Python has become particularly popular in data science and machine learning applications due to its simplicity and extensive library ecosystem. JavaScript powers modern web development, running both in browsers and on servers through Node.js. These languages offer high-level abstractions that make programming more accessible compared to the low-level machine languages of early computers. Modern integrated development environments provide powerful tools for debugging, version control, and collaborative development that were unimaginable in the era of punch cards and batch processing.""")
    ]
    
    # Create search-optimized summaries/chunks (what gets indexed)
    doc_summaries = [
        ("IBM 1401 was an early business computer system known for reliability and unique programming model."),
        ("Modern programming uses high-level languages like Python and JavaScript with powerful development tools.")
    ]
    
    # Store full documents and index summaries
    for i, (full_doc, summary) in enumerate(zip(full_docs, doc_summaries)):
        doc_id = str(uuid.uuid4())
        
        # Store full document
        docstore.mset([(doc_id, full_doc)])
        
        # Create summary document for indexing
        summary_doc = Document(
            page_content=summary,
            metadata={"doc_id": doc_id, "type": "summary"}
        )
        
        # Index the summary
        local_vectorstore.add_documents([summary_doc])
    
    print("Multi-vector setup: Indexed summaries, stored full documents...")
    print("\nQuerying with summary-optimized search...")
    
    # Query matches summaries but returns full documents
    results = retriever.invoke("early computer programming systems")
    
    print(f"Retrieved {len(results)} full documents based on summary matching:")
    for i, doc in enumerate(results, 1):
        doc_type = "Early Computing" if "IBM 1401" in doc.page_content else "Modern Programming"
        word_count = len(doc.page_content.split())
        print(f"{i}. {doc_type} ({word_count} words): {doc.page_content[:80]}...")
    
    return results

"""
8. PARENT DOCUMENT RETRIEVER
=============================
"""

def example_parent_document_retriever():
    """Example: Parent document retriever - search small chunks, return full docs"""
    from langchain.storage import InMemoryStore
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Setup components
    local_vectorstore = InMemoryVectorStore(embeddings)
    docstore = InMemoryStore()
    
    # Create text splitter for small chunks (400 chars)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # Create parent document retriever
    retriever = Retriever.create_parent_document_retriever(
        vectorstore=local_vectorstore,
        docstore=docstore,
        child_splitter=child_splitter
    )
    
    # Add long documents that will benefit from chunk-based search
    docs = [
        Document(page_content="""Machine Learning Overview: Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values). Popular algorithms include linear regression, decision trees, random forests, and support vector machines. Unsupervised learning finds hidden patterns in data without labeled examples. Clustering algorithms like k-means group similar data points together. Dimensionality reduction techniques like PCA help visualize high-dimensional data. Reinforcement learning trains agents to make sequences of decisions in an environment to maximize cumulative reward."""),
        
        Document(page_content="""Deep Learning Fundamentals: Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. Neural networks are inspired by the structure and function of biological neural networks in the brain. A basic neural network consists of an input layer, one or more hidden layers, and an output layer. Each layer contains nodes (neurons) that are connected to nodes in adjacent layers with weighted connections. During training, the network learns by adjusting these weights through backpropagation, which calculates gradients and updates weights to minimize prediction errors. Deep learning has achieved remarkable success in computer vision, natural language processing, and speech recognition. Convolutional Neural Networks (CNNs) are particularly effective for image processing tasks, while Recurrent Neural Networks (RNNs) and Transformers excel at sequential data like text and time series.""")
    ]
    
    print(f"Adding {len(docs)} long documents that will be split into small chunks...")
    retriever.add_documents(docs)
    
    # Search with specific query - will match small chunks but return full documents
    print("\nSearching for 'neural networks' - will find relevant chunks but return full documents:")
    results = retriever.invoke("neural networks and deep learning")
    
    print(f"Retrieved {len(results)} full documents:")
    for i, doc in enumerate(results, 1):
        title = "ML Overview" if "Machine Learning Overview" in doc.page_content else "Deep Learning"
        word_count = len(doc.page_content.split())
        print(f"{i}. {title} ({word_count} words): {doc.page_content[:100]}...")
    
    return results

"""
9. HYBRID SEARCH
=================
"""

async def example_hybrid_retriever():
    """Example: Hybrid search using QdrantVectorStore for true hybrid capabilities"""
    from backend.core.dependencies import get_vector_store_service
    import time
    
    collection_name = None
    vector_service = None
    
    try:
        # Try to use QdrantVectorStore for true hybrid search capabilities
        vector_service = get_vector_store_service()
        collection_name = f"hybrid_test_{int(time.time())}"
        
        # Setup travel-related content (like in docs example)
        docs = [
            Document(page_content="I visited Paris last month and saw the Eiffel Tower. The city was beautiful with amazing architecture."),
            Document(page_content="My trip to New York was incredible. I loved the museums and Broadway shows in the big apple."),
            Document(page_content="Tokyo is a fascinating city that blends traditional culture with modern technology and innovation."),
            Document(page_content="London has rich history and culture. I enjoyed visiting the museums and historic landmarks."),
            Document(page_content="Barcelona offers great food, architecture, and Mediterranean beaches. A perfect vacation destination."),
            Document(page_content="The new restaurant downtown serves excellent Italian cuisine with fresh ingredients and authentic recipes.")
        ]
        
        # Setup vectorstore with async methods
        await vector_service.add_documents(docs, collection_name=collection_name)
        vectorstore = await vector_service.get_vector_store(collection_name)
        
        # Create hybrid retriever with QdrantVectorStore
        retriever = Retriever.create_hybrid_retriever(
            vectorstore=vectorstore,
            hybrid_params={"alpha": 0.5},  # Balance between vector and keyword search
            k=3
        )
        
        print("Hybrid search example using QdrantVectorStore:")
        print("Testing with both semantic similarity and keyword matching...")
        
        # Test queries that benefit from hybrid search
        test_queries = [
            "What city did I visit last?",  # Should find "last" keyword + semantic similarity
            "amazing museums and culture",  # Should find semantic matches
            "food and restaurants"  # Should find keyword + semantic matches
        ]
        
        all_results = []
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.invoke(query)
            
            print(f"Retrieved {len(results)} documents:")
            for i, doc in enumerate(results, 1):
                # Extract city mentioned
                cities = ["Paris", "New York", "Tokyo", "London", "Barcelona"]
                city = next((c for c in cities if c.lower() in doc.page_content.lower()), "Other")
                
                # Highlight why this document was retrieved
                content_preview = doc.page_content[:60]
                if "last" in query and "last" in doc.page_content.lower():
                    print(f"{i}. {city} (keyword 'last'): {content_preview}...")
                elif any(word in doc.page_content.lower() for word in query.split()):
                    print(f"{i}. {city} (keyword match): {content_preview}...")
                else:
                    print(f"{i}. {city} (semantic): {content_preview}...")
            
            all_results.extend(results)
        
        if vectorstore.__class__.__name__ == 'QdrantVectorStore':
            print(f"\n✅ Successfully using {vectorstore.__class__.__name__} for hybrid search!")
            print("This implementation supports both dense vector similarity and keyword matching.")
        else:
            print(f"\nUsing {vectorstore.__class__.__name__} - may have limited hybrid capabilities.")
        
        return all_results
        
    except Exception as e:
        print(f"QdrantVectorStore hybrid search failed: {e}")
        print("Falling back to InMemoryVectorStore similarity search...")
        
        # Fallback to InMemoryVectorStore
        from langchain_core.vectorstores import InMemoryVectorStore
        local_vectorstore = InMemoryVectorStore(embeddings)
        docs = [
            Document(page_content="I visited Paris last month and saw the Eiffel Tower."),
            Document(page_content="My trip to New York was incredible."),
            Document(page_content="Tokyo blends traditional culture with modern technology.")
        ]
        local_vectorstore.add_documents(docs)
        
        retriever = Retriever.create_hybrid_retriever(
            vectorstore=local_vectorstore,
            hybrid_params={"alpha": 0.5},
            k=3
        )
        
        results = retriever.invoke("What city did I visit last?")
        print(f"Fallback search returned {len(results)} documents")
        print("Note: InMemoryVectorStore falls back to similarity search only.")
        
        return results
    
    finally:
        # Clean up test collection
        try:
            if collection_name and vector_service:
                vector_client = vector_service.client
                if await vector_client.collection_exists(collection_name):
                    await vector_client.delete_collection(collection_name)
                    print(f"Cleaned up test collection: {collection_name}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up hybrid test collection: {cleanup_error}")

"""
10. RETRIEVER MANAGER
======================
"""

async def example_retriever_manager():
    """Example: Managing multiple retrievers with RetrieverManager
    
    This is an async function that needs to be awaited when called.
    Usage: results = await example_retriever_manager()
    """
    # Setup
    local_vectorstore = InMemoryVectorStore(embeddings)
    docs = [Document(page_content="Example content")]
    local_vectorstore.add_documents(docs)
    
    # Create manager
    manager = RetrieverManager()
    
    # Register multiple retrievers
    manager.create_retriever(
        name="similarity",
        retriever_type="similarity",
        vectorstore=local_vectorstore,
        search_kwargs={"k": 4}
    )
    
    manager.create_retriever(
        name="mmr",
        retriever_type="mmr", 
        vectorstore=local_vectorstore,
        search_kwargs={"k": 4, "fetch_k": 20}
    )
    
    # Use retrievers by name
    similarity_results = await manager.retrieve("similarity", "query text")
    mmr_results = await manager.retrieve("mmr", "query text")
    
    return similarity_results, mmr_results

def example_retriever_manager_sync():
    """Example: Synchronous version of retriever manager
    
    For non-async usage, use the retrievers directly.
    """
    # Setup
    local_vectorstore = InMemoryVectorStore(embeddings)
    docs = [Document(page_content="Example content")]
    local_vectorstore.add_documents(docs)
    
    # Create retrievers directly for sync usage
    similarity_retriever = Retriever.create_vector_retriever(
        vectorstore=local_vectorstore,
        search_type="similarity",
        k=4
    )
    
    mmr_retriever = Retriever.create_vector_retriever(
        vectorstore=local_vectorstore,
        search_type="mmr",
        k=4,
        fetch_k=20
    )
    
    # Use retrievers directly (synchronous)
    similarity_results = similarity_retriever.invoke("query text")
    mmr_results = mmr_retriever.invoke("query text")
    
    return similarity_results, mmr_results


async def main():
    """Run all retriever tests."""
    print("=== Running Retriever Tests ===\n")
    
    # 1. Vector Store Retrievers
    print("1. VECTOR STORE RETRIEVERS")
    print("=" * 30)
    
    print("Testing similarity retriever...")
    example_similarity_retriever()
    print()
    
    print("Testing MMR retriever...")
    example_mmr_retriever()
    print()
    
    print("Testing score retriever...")
    example_score_retriever()
    print()
    
    print("Testing threshold retriever...")
    example_threshold_retriever()
    print()
    
    # 2. Multi-Query Retriever
    print("2. MULTI-QUERY RETRIEVER")
    print("=" * 25)
    
    print("Testing multi-query retriever...")
    example_multi_query_retriever()
    print()
    
    # 3. Contextual Compression
    print("3. CONTEXTUAL COMPRESSION")
    print("=" * 26)
    
    print("Testing LLM compression retriever...")
    example_llm_compression_retriever()
    print()
    
    print("Testing embeddings compression retriever...")
    example_embeddings_compression_retriever()
    print()
    
    # 4. Ensemble Retriever
    print("4. ENSEMBLE RETRIEVER")
    print("=" * 21)
    
    print("Testing ensemble retriever...")
    example_ensemble_retriever()
    print()
    
    # 5. Long Context Reorder
    print("5. LONG CONTEXT REORDER")
    print("=" * 23)
    
    print("Testing reorder retriever...")
    example_reorder_retriever()
    print()
    
    # 6. Self-Query Retriever
    print("6. SELF-QUERY RETRIEVER")
    print("=" * 23)
    
    print("Testing self-query retriever...")
    await example_self_query_retriever()
    print()
    
    # 7. Multi-Vector Retriever
    print("7. MULTI-VECTOR RETRIEVER")
    print("=" * 25)
    
    print("Testing multi-vector retriever...")
    example_multi_vector_retriever()
    print()
    
    # 8. Parent Document Retriever
    print("8. PARENT DOCUMENT RETRIEVER")
    print("=" * 28)
    
    print("Testing parent document retriever...")
    example_parent_document_retriever()
    print()
    
    # 9. Hybrid Search
    print("9. HYBRID SEARCH")
    print("=" * 16)
    
    print("Testing hybrid retriever...")
    await example_hybrid_retriever()
    print()
    
    # 10. Retriever Manager
    print("10. RETRIEVER MANAGER")
    print("=" * 21)
    
    print("Testing retriever manager (async)...")
    await example_retriever_manager()
    print()
    
    print("Testing retriever manager (sync)...")
    example_retriever_manager_sync()
    print()
    
    print("=== All retriever tests completed ===")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

"""
USAGE NOTES:
============

1. All retrievers inherit from LangChain's BaseRetriever interface
2. Use .invoke(query) for synchronous retrieval
3. Use .ainvoke(query) for asynchronous retrieval  
4. Configure retrievers through search_kwargs parameter
5. Composite retrievers (ensemble, multi-query) require base retrievers
6. Score retrievers add similarity scores to document metadata
7. Compression retrievers reduce document content while preserving relevance
8. Self-query retrievers enable natural language metadata filtering
9. Multi-vector and parent document retrievers handle complex document relationships
10. RetrieverManager enables centralized retriever management
11. Use .add_documents() for parent_document and multi_vector retriever types
12. Async functions need to be awaited: results = await example_retriever_manager()

VECTORSTORE COMPATIBILITY:
==========================

- InMemoryVectorStore: Supports most retriever types except self-query
- Self-query supported vectorstores: Chroma, Pinecone, Weaviate, Qdrant, FAISS, ElasticSearch
- For InMemoryVectorStore with metadata filtering, use manual filtering approach
- Hybrid search functionality depends on specific vectorstore implementation

ERROR HANDLING:
===============

All retriever methods include proper validation and error handling:
- Vectorstore validation for vector-based operations
- Required parameter validation for composite retrievers
- Graceful fallbacks for optional dependencies
- Clear error messages for misconfiguration

PERFORMANCE CONSIDERATIONS:
===========================

- Use appropriate chunk sizes for parent document retriever
- Configure similarity thresholds to filter irrelevant results
- Balance weights in ensemble retrievers based on use case
- Consider computational cost of LLM-based compression
- Use MMR for diverse results when avoiding redundancy
"""
    
