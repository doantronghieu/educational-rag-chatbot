"""
Test document collections for retriever examples.

These shared document collections reduce redundancy across examples
and provide realistic, diverse content for testing different retriever types.
"""

from langchain_core.documents import Document


def get_programming_docs():
    """Programming and technology documents with varying relevance levels"""
    return [
        Document(page_content="Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and machine learning."),
        Document(page_content="JavaScript is the programming language of the web. It enables interactive web pages and is an essential part of web applications alongside HTML and CSS."),
        Document(page_content="Machine learning is a method of data analysis that automates analytical model building. It uses algorithms that iteratively learn from data."),
        Document(page_content="Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract insights from data."),
        Document(page_content="Web development involves creating websites and web applications. It includes frontend development with HTML, CSS, and JavaScript."),
        Document(page_content="Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn.")
    ]


def get_ml_ai_docs():
    """Machine learning and AI documents with clear relevance differences"""
    return [
        Document(page_content="Machine learning algorithms automatically learn patterns from data to make predictions and decisions."),
        Document(page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers."),
        Document(page_content="Artificial intelligence encompasses machine learning and other techniques to create intelligent systems."),
        Document(page_content="Data science involves extracting insights from data using statistical methods and programming."),
        Document(page_content="Web development focuses on creating websites and web applications using various technologies."),
        Document(page_content="Mobile app development involves creating applications for smartphones and tablets.")
    ]


def get_cloud_computing_docs():
    """Cloud computing documents with mixed relevance for threshold testing"""
    return [
        Document(page_content="Cloud computing provides scalable infrastructure and services over the internet for businesses."),
        Document(page_content="Amazon Web Services (AWS) offers comprehensive cloud computing platforms and APIs."),
        Document(page_content="Microsoft Azure is a cloud computing service for building, testing, and managing applications."),
        Document(page_content="Serverless computing allows developers to build applications without managing server infrastructure."),
        Document(page_content="The weather forecast predicts rain and thunderstorms for the weekend."),
        Document(page_content="Traditional cooking methods have been passed down through generations of families."),
        Document(page_content="Digital transformation helps businesses modernize their operations using cloud technologies.")
    ]


def get_python_overlap_docs():
    """Python documents with intentional overlap for MMR demonstration"""
    return [
        Document(page_content="Python is an interpreted, high-level programming language with dynamic semantics. Python's simple syntax emphasizes readability."),
        Document(page_content="Python programming language is known for its clean, readable syntax. Python supports multiple programming paradigms."),
        Document(page_content="Python code is easy to read and write. The Python language emphasizes code readability with its notable use of whitespace."),
        Document(page_content="Java is a class-based, object-oriented programming language. Java applications are compiled to bytecode."),
        Document(page_content="JavaScript is a high-level programming language. JavaScript is the programming language of the World Wide Web."),
        Document(page_content="C++ is a general-purpose programming language. C++ was originally developed as an extension of the C language."),
        Document(page_content="Machine learning algorithms can learn patterns from data without being explicitly programmed for each task."),
        Document(page_content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.")
    ]


def get_apple_docs():
    """Apple fruit vs Apple Inc. documents for ensemble retrieval demonstration"""
    return [
        Document(page_content="Fresh apples are delicious fruits rich in vitamins and fiber. Red apples and green apples both offer great nutritional value."),
        Document(page_content="Apple Inc. is a technology company known for innovative products like the iPhone, iPad, and Mac computers."),
        Document(page_content="Fruit orchards grow many varieties of apples including Granny Smith, Red Delicious, and Honeycrisp apples."),
        Document(page_content="The apple tree is a deciduous tree in the rose family best known for its sweet, pomaceous fruit, the apple."),
        Document(page_content="Steve Jobs co-founded Apple Computer Inc. and helped transform it into the world's most valuable technology company."),
        Document(page_content="Nutritionists recommend eating apples daily as they contain antioxidants and promote digestive health."),
        Document(page_content="Technology stocks like Apple, Google, and Microsoft have driven market growth in recent years."),
        Document(page_content="Orchard management involves pruning apple trees, pest control, and harvesting at optimal ripeness.")
    ]


def get_task_decomposition_docs():
    """Task decomposition and planning documents for multi-query examples"""
    return [
        Document(page_content="Task decomposition involves breaking down complex problems into smaller, manageable subtasks that can be solved independently."),
        Document(page_content="Planning in AI systems requires decomposing high-level goals into actionable steps with clear dependencies and resource requirements."),
        Document(page_content="Divide and conquer is a classic algorithmic approach that breaks problems into smaller subproblems of the same type."),
        Document(page_content="Project management methodologies like Agile break large projects into smaller sprints and user stories."),
        Document(page_content="Hierarchical task networks represent complex tasks as trees of subtasks with preconditions and effects."),
        Document(page_content="Workflow automation systems decompose business processes into discrete, executable steps."),
        Document(page_content="Problem-solving strategies often involve analyzing complex situations by examining their component parts."),
        Document(page_content="Systems thinking approaches complex problems by understanding the relationships between different components.")
    ]


def get_celtics_docs():
    """Boston Celtics documents with irrelevant content for reordering demonstration"""
    return [
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


def get_movie_docs():
    """Movie documents with rich metadata for self-query examples"""
    return [
        Document(
            page_content="Interstellar explores humanity's journey through space and time to find a new home for Earth's population.",
            metadata={"genre": "sci-fi", "year": 2014, "rating": 8.6, "director": "Christopher Nolan"}
        ),
        Document(
            page_content="The Princess Bride is a classic fantasy adventure with romance, comedy, and sword fighting.",
            metadata={"genre": "fantasy", "year": 1987, "rating": 8.0, "director": "Rob Reiner"}
        ),
        Document(
            page_content="Blade Runner 2049 is a neo-noir science fiction film set in a dystopian future.",
            metadata={"genre": "sci-fi", "year": 2017, "rating": 8.0, "director": "Denis Villeneuve"}
        ),
        Document(
            page_content="The Grand Budapest Hotel is a comedy-drama about the adventures of a legendary concierge.",
            metadata={"genre": "comedy", "year": 2014, "rating": 8.1, "director": "Wes Anderson"}
        ),
        Document(
            page_content="Dune is an epic science fiction film about power, politics, and survival on a desert planet.",
            metadata={"genre": "sci-fi", "year": 2021, "rating": 8.0, "director": "Denis Villeneuve"}
        ),
        Document(
            page_content="Mad Max: Fury Road is a high-octane action film set in a post-apocalyptic wasteland.",
            metadata={"genre": "action", "year": 2015, "rating": 8.1, "director": "George Miller"}
        )
    ]


def get_travel_docs():
    """Travel documents for hybrid search examples"""
    return [
        Document(page_content="I visited Paris last month and saw the Eiffel Tower. The city was beautiful with amazing architecture."),
        Document(page_content="My trip to New York was incredible. I loved the museums and Broadway shows in the big apple."),
        Document(page_content="Tokyo is a fascinating city that blends traditional culture with modern technology and innovation."),
        Document(page_content="London has rich history and culture. I enjoyed visiting the museums and historic landmarks."),
        Document(page_content="Barcelona offers great food, architecture, and Mediterranean beaches. A perfect vacation destination."),
        Document(page_content="The new restaurant downtown serves excellent Italian cuisine with fresh ingredients and authentic recipes.")
    ]


def get_compression_docs():
    """Long documents for compression examples"""
    return [
        Document(page_content="""Judge Ketanji Brown Jackson was nominated to the Supreme Court by President Biden in February 2022. She previously served on the U.S. Court of Appeals for the D.C. Circuit and the U.S. District Court for the District of Columbia. Before her judicial career, Jackson worked as a federal public defender and was a partner at a law firm. She graduated from Harvard Law School where she served as an editor of the Harvard Law Review. Jackson is known for her thorough preparation and thoughtful questioning during hearings. The nomination process involved extensive Senate hearings where she answered questions about her judicial philosophy, past decisions, and approach to constitutional interpretation. Her confirmation made her the first Black woman to serve on the Supreme Court."""),
        
        Document(page_content="""The Supreme Court of the United States consists of nine justices who serve life tenure. The Court's primary function is to interpret the Constitution and federal law. Cases reach the Court through a writ of certiorari, where the justices vote on which cases to hear. The Court typically hears 60-80 cases per year out of thousands of petitions. Justices write majority opinions, concurring opinions, and dissenting opinions to explain their reasoning. The Court's decisions are binding on all lower courts and establish legal precedent. Famous landmark cases include Brown v. Board of Education, Roe v. Wade, and Miranda v. Arizona. The building where the Court sits was completed in 1935 and features neoclassical architecture. Court sessions run from October through June, with summer recess for writing opinions.""")
    ]


def get_ai_filter_docs():
    """AI/ML documents with mixed relevance for embeddings filtering"""
    return [
        Document(page_content="Artificial intelligence and machine learning are transforming modern technology and business applications."),
        Document(page_content="Deep learning neural networks use multiple layers to process complex patterns in data and images."),
        Document(page_content="Natural language processing enables computers to understand and generate human language effectively."),
        Document(page_content="The weather today is sunny and perfect for outdoor activities like hiking and picnics."),
        Document(page_content="Cooking delicious pasta requires proper timing, quality ingredients, and attention to detail."),
        Document(page_content="Computer vision algorithms can identify objects, faces, and scenes in digital images and videos."),
        Document(page_content="Music streaming services have changed how people discover and listen to their favorite songs."),
        Document(page_content="Data science combines statistics, programming, and domain knowledge to extract insights from data.")
    ]


def get_long_ml_docs():
    """Long ML/computing documents for parent document and multi-vector examples"""
    return [
        Document(page_content="""Machine Learning Overview: Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values). Popular algorithms include linear regression, decision trees, random forests, and support vector machines. Unsupervised learning finds hidden patterns in data without labeled examples. Clustering algorithms like k-means group similar data points together. Dimensionality reduction techniques like PCA help visualize high-dimensional data. Reinforcement learning trains agents to make sequences of decisions in an environment to maximize cumulative reward."""),
        
        Document(page_content="""Deep Learning Fundamentals: Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. Neural networks are inspired by the structure and function of biological neural networks in the brain. A basic neural network consists of an input layer, one or more hidden layers, and an output layer. Each layer contains nodes (neurons) that are connected to nodes in adjacent layers with weighted connections. During training, the network learns by adjusting these weights through backpropagation, which calculates gradients and updates weights to minimize prediction errors. Deep learning has achieved remarkable success in computer vision, natural language processing, and speech recognition. Convolutional Neural Networks (CNNs) are particularly effective for image processing tasks, while Recurrent Neural Networks (RNNs) and Transformers excel at sequential data like text and time series.""")
    ]


def get_computing_history_docs():
    """Computing history documents for multi-vector examples"""
    return [
        Document(page_content="""The IBM 1401 Computer System: In the early days of computing, the IBM 1401 was a variable word length decimal computer that was announced by IBM on October 5, 1959. The system was widely used for business applications and was known for its reliability and ease of programming. Programming the IBM 1401 required understanding its unique instruction set and memory organization. The computer used a magnetic core memory system and supported various input/output devices including card readers, line printers, and magnetic tape drives. Many programmers got their start on systems like the IBM 1401, learning fundamental concepts that would serve them throughout their careers in computing."""),
        
        Document(page_content="""Modern Programming Languages: Today's programming landscape is dominated by languages like Python, JavaScript, and Java. Python has become particularly popular in data science and machine learning applications due to its simplicity and extensive library ecosystem. JavaScript powers modern web development, running both in browsers and on servers through Node.js. These languages offer high-level abstractions that make programming more accessible compared to the low-level machine languages of early computers. Modern integrated development environments provide powerful tools for debugging, version control, and collaborative development that were unimaginable in the era of punch cards and batch processing.""")
    ]