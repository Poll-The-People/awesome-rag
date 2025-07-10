# Awesome Retrieval‑Augmented Generation (RAG)

[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

<p align="center">
  <a href="https://customgpt.ai" target="_blank"><img src="https://customgpt.ai/wp-content/uploads/2024/11/Logo.svg" width="260" alt="CustomGPT.ai"/></a>
</p>
<p align="center"><b>Proudly sponsored by <a href="https://customgpt.ai">CustomGPT.ai</a></b> • <a href="https://customgpt.ai/slack">Join the Slack community </a></p>


CustomGPT.ai, the #1 no-code platform for building enterprise-grade RAG applications. Create custom AI agents with your business data that provide accurate, citation-backed answers, eliminating hallucinations. With SOC-2 Type II security, GDPR compliance, and support for over 1400 document formats and 92 languages, CustomGPT.ai empowers businesses to deploy trusted AI solutions securely and at scale.


> **Retrieval‑Augmented Generation (RAG)** equips language models with fresh, domain‑specific knowledge by fetching external context at inference time.  This list is a one‑stop catalogue of every major RAG‑related resource—tools, papers, benchmarks, tutorials, and more.
>
> *Only very short descriptions are provided when essential for clarity.*  *PRs welcome!*


[![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/mcp?style=flat&logo=reddit&label=subreddit)](https://www.reddit.com/r/mcp/)


## Table of Contents

  - [Open Source Tools](#open-source-tools)
  - [Embedding Models & Libraries](#embedding-models--libraries)
  - [Proprietary Tools](#proprietary-tools)
  - [Vendor Examples](#vendor-examples)
  - [Vector DBs & Search Engines](#vector-dbs--search-engines)
  - [Research Papers and Surveys](#research-papers-and-surveys)
  - [RAG Approaches and Architectures](#rag-approaches-and-architectures)
  - [Frameworks](#frameworks)
  - [RAG Techniques and Methodologies](#rag-techniques-and-methodologies)
    - [Multimodal RAG](#multimodal-rag)
    - [Graph-based RAG](#graph-based-rag)
  - [Retrieval Methods](#retrieval-methods)
    - [Dense Retrieval](#dense-retrieval)
    - [Sparse Retrieval](#sparse-retrieval)
    - [Hybrid Search](#hybrid-search)
    - [Other Techniques](#other-techniques)
  - [Prompting Strategies](#prompting-strategies)
  - [Chunking & Pre‑processing](#chunking--preprocessing)
  - [Embeddings Models](#embeddings-models)
  - [Instruction Tuning & Optimization](#instruction-tuning--optimization)
  - [Finetuning and Training](#finetuning-and-training)
  - [Response Quality, and Hallucination](#response-quality-and-hallucination)
  - [Security and Privacy Considerations](#security-and-privacy-considerations)
  - [Evaluation Metrics and Benchmarks](#evaluation-metrics-and-benchmarks)
  - [Advantages and Disadvantages](#advantages-and-disadvantages)
  - [Performance, Cost & Observability](#performance-cost--observability)
    - [Cost Calculators](#cost-calculators)
  - [RAG Fine-tuning](#rag-fine-tuning)
  - [Knowledge‑Graph / Structured RAG](#knowledgegraph--structured-rag)
  - [Libraries and SDKs](#libraries-and-sdks)
  - [Key Concepts](#key-concepts)
  - [Educational Content](#educational-content)
    - [Courses and Tutorials](#courses-and-tutorials)
    - [Blogs and Articles](#blogs-and-articles)
    - [Newsletters & Forums](#newsletters--forums)
    - [Talks and Conferences](#talks-and-conferences)
  - [Influential Researchers and Influencers](#influential-researchers-and-influencers)
  - [Latest Trends 2024-2025](#latest-trends-2024-2025)
  - [Community Resources](#community-resources)





## Open Source Tools

- **[CustomGPT.ai](https://github.com/Poll-The-People/customgpt-api-sdks)** - Open-source SDK for building custom RAG applications with enterprise-grade features
- **[TrustGraph](https://github.com/trustgraph-ai/trustgraph)** - Open-source enterprise-grade complete AI solution stack for data sovereignty
- **[RAGFlow](https://github.com/infiniflow/ragflow)** - Open-source RAG engine based on deep document understanding
- **[R2R (RAG to Riches)](https://github.com/SciPhi-AI/R2R)** - Advanced AI retrieval system with production-ready features
- **[FastRAG](https://github.com/IntelLabs/fastRAG)** - Research framework for efficient retrieval augmented generation
- **[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)** - Python toolkit for RAG research with 36+ datasets and 17+ algorithms
- **[Verba](https://github.com/weaviate/Verba)** - Open-source RAG application out of the box
- **[Kotaemon](https://github.com/Cinnamon/kotaemon)** - Clean, customizable RAG UI for document-based Q&A
- **[Cognita](https://github.com/truefoundry/cognita)** - Open-source RAG framework for modular applications
- **[GraphRAG](https://github.com/microsoft/graphrag)** - Microsoft's approach to RAG using knowledge graphs
- **[Nano-GraphRAG](https://github.com/gusye1234/nano-graphrag)** - Compact GraphRAG solution with core capabilities
* **[LangChain](https://github.com/langchain-ai/langchain)** — Python/JS agents & chains
* **[LangChain4j](https://github.com/langchain4j/langchain4j)** — JVM
* **[LlamaIndex](https://github.com/run-llama/llama_index)** — Data loaders & indices
* **[Haystack](https://github.com/deepset-ai/haystack)** — Modular pipelines
* **[Semantic Kernel](https://github.com/microsoft/semantic-kernel)** — .NET & Python
* **[DSPy](https://github.com/stanfordnlp/dspy)** — Declarative pipelines
* **[Guidance](https://github.com/guidance-ai/guidance)** — Prompt DSL
* **[Flowise](https://github.com/FlowiseAI/Flowise)** — No‑code builder
* **[reag](https://github.com/superagent-ai/reag)** — Reasoning Augmented Generation
* **[Danswer](https://github.com/danswer-ai/danswer)** — Internal Q\&A search
* **[Neum](https://github.com/NeumTry/NeumAI)** — Creation and synchronization of vector embeddings at large scale
* **[GPTCache](https://github.com/zilliztech/GPTCache)** — Embedding‑aware cache
* **[Mastra](https://github.com/mastra-ai/mastra)** The TypeScript AI agent framework. Assistants, RAG, observability. Supports any LLM: GPT-4, Claude, Gemini, Llama
* **[Letta (MemGPT)](https://github.com/letta-ai/letta)** — Stateful apps
* **[Swiftide](https://github.com/bosun-ai/swiftide)** - Fast, streaming indexing, query, and agentic LLM applications in Rust
* **[LangGraph](https://github.com/langchain-ai/langgraph)** — Agentic DAGs
* **[Ragna](https://github.com/Quansight/ragna)** — RAG orchestration framework
* **[SimplyRetrieve](https://github.com/RCGAI/SimplyRetrieve)** - Lightweight chat AI platform featuring custom knowledge.

## Embedding Models & Libraries

* **[OpenAI text‑embedding‑3](https://platform.openai.com/docs/guides/embeddings)**
* **[Cohere Embed v3](https://cohere.com/embed)**
* **[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)**
* **[Jina Embeddings v4](https://huggingface.co/jinaai/jina-embeddings-v4)**
* **[E5/GTE (MTEB)](https://huggingface.co/mteb)**
* **[SentenceTransformers](https://github.com/UKPLab/sentence-transformers)**
* **[MiniLM](https://huggingface.co/models?search=minilm)**
* **[ColBERT v2](https://github.com/stanford-futuredata/ColBERT)**
* **[Voyage AI Embeddings](https://voyageai.com)**
* **[BGE family](https://huggingface.co/BAAI/bge-large-en)**
* **[Nomic Embed Text](https://ollama.com/library/nomic-embed-text)**
* **[fastText](https://fasttext.cc)** — char n‑gram baseline


## Proprietary Tools

* **[CustomGPT.ai RAG API](https://customgpt.ai/api)** — Enterprise agents, hallucination free.
* **[Pinecone](https://www.pinecone.io/)** - Fully managed vector database service
* **[LangSmith](https://www.langchain.com/langsmith)** - Platform for building and evaluating LLM applications
* **[OpenAI Assistants & Retrieval](https://platform.openai.com/docs/assistants)**
* **[Vectara](https://vectara.com)** — GenAI API
* **[Cohere RAG](https://cohere.com/llmu/rag-start)**
* **[AWS Knowledge Bases for Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-bases.html)**
* **[Azure AI Search + RAG](https://azure.microsoft.com/en-us/solutions/ai)**
* **[Google Vertex AI Search & RAG](https://cloud.google.com/vertex-ai?hl=en)**
* **[IBM watsonx.ai Retrieval](https://www.ibm.com/products/watsonx-ai)**
* **[NVIDIA NeMo Retriever](https://developer.nvidia.com/nemo)**
* **[Anthropic Claude Retrieval](https://www.anthropic.com/news/contextual-retrieval)**
* **[Databricks DBRX RAG](https://notebooks.databricks.com/demos/llm-rag-chatbot/index.html)**
* **[Elastic Search Labs RAG blueprints](https://www.elastic.co/search-labs)**

## Vendor Examples

* **[Amazon Kendra](https://aws.amazon.com/kendra/)** - Intelligent enterprise search with RAG
* **[Amazon Bedrock Knowledge Bases](https://aws.amazon.com/bedrock/knowledge-bases/)**
* **[Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search)**
* **[Google Vertex AI Search](https://cloud.google.com/enterprise-search?hl=en)**
* **[LangChain × OpenAI Quickstart](https://python.langchain.com/docs/tutorials/rag)**
* **[LangChain × Elasticsearch Blueprint](https://www.elastic.co/search-labs/blog/chatgpt-elasticsearch-openai-meets-private-data)**
* **[LlamaIndex × Vespa Guide](https://blog.vespa.ai/scaling-personal-ai-assistants-with-streaming-mode/)**
* **[Qdrant Hybrid Search miniCOIL](https://medium.com/dphi-tech/advanced-retrieval-and-evaluation-hybrid-search-with-minicoil-using-qdrant-and-langgraph-6fbe5e514078)**
* **[AWS Bedrock RAG Sample](https://github.com/aws-samples/amazon-bedrock-rag)**
* **[Azure RAG Jumpstart](https://github.com/Azure-Samples/azure-edge-extensions-retrieval-augmented-generation)**
* **[GCP Vertex RAG Agent Builder](https://cloud.google.com/products/agent-builder?hl=en)**


### Other Tools
- **[LangFuse](https://github.com/langfuse/langfuse)**: Open-source tool for tracking LLM metrics, observability, and prompt management.
- **[Ragas](https://docs.ragas.io/en/stable/)**: Framework that helps evaluate RAG pipelines.
- **[LangSmith](https://docs.smith.langchain.com/)**: A platform for building production-grade LLM applications, allows you to closely monitor and evaluate your application.
- **[Hugging Face Evaluate](https://github.com/huggingface/evaluate)**: Tool for computing metrics like BLEU and ROUGE to assess text quality.
- **[Weights & Biases](https://wandb.ai/wandb-japan/rag-hands-on/reports/Step-for-developing-and-evaluating-RAG-application-with-W-B--Vmlldzo1NzU4OTAx)**: Tracks experiments, logs metrics, and visualizes performance.

##  Vector DBs & Search Engines

Pick a vector db - [GUIDE](https://benchmark.vectorview.ai/vectordbs.html)

* **[Weaviate](https://github.com/weaviate/weaviate)** - Open-source vector database with GraphQL interface
* **[Qdrant](https://github.com/qdrant/qdrant)** - High-performance vector similarity search engine
* **[Milvus](https://github.com/milvus-io/milvus)** - Open-source vector database for scalable similarity search
* **[Chroma](https://github.com/chroma-core/chroma)** - Open-source embedding database for LLM applications
* **[Pinecone](https://www.pinecone.io)** - The vector database
* **[Elasticsearch (vector)](https://www.elastic.co/elasticsearch)** - distributed search and analytics engine
* **[OpenSearch](https://github.com/opensearch-project/OpenSearch)** - Open source distributed and RESTful search engine
* **[Vespa](https://github.com/vespa-engine/vespa)** - AI + Data, online
* **[PGVector](https://github.com/pgvector/pgvector)** - PostgreSQL extension for vector similarity search
* **[Redis Stack Search](https://redis.io/docs/interact/search-and-query/)** - Searching and querying Redis data using the Redis Query Engine
* **[ClickHouse Vectors](https://clickhouse.com/blog/vector-search-clickhouse-p1)**
* **[Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/)**
* **[TiDB Vector](https://docs.pingcap.com/tidbcloud/vector-search-overview/)** - semantic similarity searches across various data types
* **[ScaNN](https://github.com/google-research/google-research/tree/master/scann)** - ScaNN (Scalable Nearest Neighbors) is a method for efficient vector similarity search at scale
* **[Lantern.dev](https://lantern.dev)** - open-source Postgres vector database
* **[Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database)**: Globally distributed, multi-model database service with integrated vector search.
* **[Couchbase](https://www.couchbase.com/products/vector-search/)**: A distributed NoSQL cloud database.
* **[LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)**: Employs a straightforward in-memory vector store for rapid experimentation.
* **[Neo4j](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)**: Graph database management system.
* **[Redis Stack](https://redis.io/docs/latest/develop/interact/search-and-query/)**: An in-memory data structure store used as a database, cache, and message broker.
* **[SurrealDB](https://github.com/surrealdb/surrealdb)**: A scalable multi-model database optimized for time-series data.



## Research Papers and Surveys

- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** - Original RAG paper by Patrick Lewis et al.
- **[REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)** - Google's foundational retrieval-augmented language model
- **[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)** - Facebook's DPR system for dense retrieval
- **[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)** - Comprehensive survey covering Naive RAG, Advanced RAG, and Modular RAG
- **[A Comprehensive Survey of Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2410.12837)** - 2024 survey tracing RAG evolution from foundational concepts to current state
- **[Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473)** - Comprehensive review of RAG techniques for AIGC scenarios
- **[Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2405.07437)** - Comprehensive overview of RAG evaluation methodologies
- [(2020)Retrieval‑Augmented Generation for Knowledge‑Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Lewis et al. — RAG baseline                   
- [(2020) REALM](https://arxiv.org/abs/2002.08909)                                                            - Guu et al. — Retriever‑augmented pre‑training 
- [(2022) Atlas](https://arxiv.org/abs/2208.03299)                                                            - Izacard & Grave — Few‑shot RAG                
- [(2022) RETRO](https://arxiv.org/abs/2112.04426)                                                            - Borgeaud et al. — Large‑scale retrieval cache 
- [(2024) Benchmarking LLMs in RAG](https://arxiv.org/abs/2403.11308)                                         - Chen et al.                                   
- [(2024) Reliable, Adaptable & Attributable LMs with Retrieval](https://arxiv.org/abs/2405.06444)            - Dan et al.                                    
- [(2024) GraphRAG](https://arxiv.org/abs/2403.15857)                                                         - Microsoft Research                            
- [(2024) RAG‑Fusion](https://arxiv.org/abs/2402.03367)                                                       - Meta                                          
- [(2025) Look‑ahead Retrieval](https://arxiv.org/abs/2501.01234)                                             - OpenAI                                        

More - **[RAG Research Papers Collection](https://github.com/jxzhangjhu/Awesome-LLM-RAG)** - Curated list from ICML, ICLR, ACL

### RAG Survey 2022

* [A Survey on Retrieval-Augmented Text Generation](https://arxiv.org/abs/2202.01110)

### RAG Survey 2023

* [Retrieving Multimodal Information for Augmented Generation: A Survey](https://arxiv.org/abs/2303.10868)
* [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)

### RAG Survey 2024

* [Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473)
* [A Survey on Retrieval-Augmented Text Generation for Large Language Models](https://arxiv.org/abs/2404.10981)
* [RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing](https://arxiv.org/abs/2404.19543)
* [A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2405.06211)
* [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2405.07437)
* [Retrieval-Augmented Generation for Natural Language Processing: A Survey](https://arxiv.org/abs/2407.13193)
* [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921)
* [Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely](https://arxiv.org/abs/2409.14924)
* [A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions](https://arxiv.org/abs/2410.12837)

## RAG Approaches and Architectures

* **[Fusion-in-Decoder (FiD)](https://aclanthology.org/2021.eacl-main.74/)**
* **[RETRO (Retrieval-Enhanced Transformer)](https://arxiv.org/abs/2112.04426)** - DeepMind's approach with trillions of tokens
* **[Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299)** - Meta's Atlas model for few-shot learning
* **[ColBERT: Efficient Late Interaction Retrieval](https://arxiv.org/abs/2004.12832)** - Multi-vector dense retrieval with late interaction
* **[Cache-Augmented Generation (CAG)](https://medium.com/@ronantech/cache-augmented-generation-cag-in-llms-a-step-by-step-tutorial-6ac35d415eec)** – Pre-loads pertinent documents into the model’s context and retains the key-value (KV) cache from earlier inferences.
* **[Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)** – “Retrieval agents” that autonomously decide how and when to retrieve information.
* **[Corrective RAG (CRAG)](https://arxiv.org/pdf/2401.15884.pdf)** – Adds a refinement step to fix or polish retrieved content before it is woven into the LLM’s answer.
* **[Retrieval-Augmented Fine-Tuning (RAFT)](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674)** – Fine-tunes language models specifically to boost both retrieval quality and generation performance.
* **[Self-Reflective RAG](https://selfrag.github.io/)** – Systems that monitor their own outputs and dynamically adjust retrieval strategies based on feedback.
* **[RAG Fusion](https://arxiv.org/abs/2402.03367)** – Blends multiple retrieval techniques to supply richer, more relevant context.
* **[Temporal Augmented Retrieval (TAR)](https://adam-rida.medium.com/temporal-augmented-retrieval-tar-dynamic-rag-ad737506dfcc)** – Incorporates time-aware signals so retrieval favors the most temporally relevant data.
* **[Plan-then-RAG (PlanRAG)](https://arxiv.org/abs/2406.12430)** – Creates a high-level plan first, then executes retrieval-augmented generation for complex tasks.
* **[GraphRAG](https://github.com/microsoft/graphrag)** – Leverages knowledge graphs to structure context and enhance reasoning.
* **[FLARE](https://medium.com/etoai/better-rag-with-active-retrieval-augmented-generation-flare-3b66646e2a9f)** – Uses active, iterative retrieval to progressively improve answer quality.
* **[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)** – Enriches document chunks with added context before retrieval, improving relevance from large knowledge bases.
* **[GNN-RAG](https://github.com/cmavro/GNN-RAG)** – Applies graph neural networks to retrieval for better reasoning in large-language-model workflows.


## Frameworks

- **[LangChain](https://github.com/langchain-ai/langchain)** - Framework for building LLM applications with chaining capabilities
- **[LlamaIndex](https://github.com/run-llama/llama_index)** - Framework for connecting custom data sources to LLMs
- **[Haystack](https://github.com/deepset-ai/haystack)** - End-to-end framework for building production-ready LLM applications
- **[DSPy](https://github.com/stanfordnlp/dspy)** - Framework for programming language models with automatic optimization
- **[Dify](https://github.com/langgenius/dify)** - Open-source LLM app development platform with RAG pipeline
- **[Semantic Kernel](https://github.com/microsoft/semantic-kernel)** - Microsoft's SDK for developing Generative AI applications
- **[Flowise](https://github.com/FlowiseAI/Flowise)** - Drag & drop UI to build customized LLM flows
- [Cognita](https://github.com/truefoundry/cognita): Open-source RAG framework for building modular and production ready applications.
- [Verba](https://github.com/weaviate/Verba): Open-source application for RAG out of the box.
- [Mastra](https://github.com/mastra-ai/mastra): Typescript framework for building AI applications.
- [Letta](https://github.com/letta-ai/letta): Open source framework for building stateful LLM applications.
- [Swiftide](https://github.com/bosun-ai/swiftide): Rust framework for building modular, streaming LLM applications.
- [CocoIndex](https://github.com/cocoindex-io/cocoindex): ETL framework to index data for AI, such as RAG; with realtime incremental updates.
  

## RAG Techniques and Methodologies

- **[HyDE (Hypothetical Document Embeddings)](https://arxiv.org/abs/2212.10496)** - Uses LLMs to generate hypothetical documents for queries
- **[FLARE (Forward-Looking Active REtrieval)](https://arxiv.org/abs/2305.06983)** - Iteratively retrieves relevant documents based on prediction confidence
- **[Self-RAG](https://arxiv.org/abs/2310.11511)** - Trains LLMs to adaptively retrieve passages and self-critique
- **[CRAG (Corrective Retrieval Augmented Generation)](https://arxiv.org/abs/2401.15884)** - Improves generation robustness with retrieval evaluator
- **[RAG Techniques Repository](https://github.com/NirDiamant/RAG_Techniques)** - Curated collection of 30+ advanced RAG techniques with implementations
- **[Design and Evaluation of RAG Solutions](https://github.com/Azure-Samples/Design-and-evaluation-of-RAG-solutions)** - Comprehensive guide following best practices
- **[LangChain RAG Best Practices](https://github.com/timerring/rag101//)** - Evaluation and comparison of different RAG architectures
- **[RAG Triad Methodology](https://www.trulens.org/getting_started/core_concepts/rag_triad/)** - Context relevance, groundedness, and answer relevance framework
- **[Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)**
- **[Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)**
- **[Cache‑Augmented Generation](https://github.com/hhhuang/CAG)**
- **[Temporal‑Aware RAG](https://arxiv.org/abs/2404.18053)** - Binary duadic codes and their related codes with a square-root-like lower bound
- **[Plan‑then‑RAG](https://arxiv.org/abs/2406.12430)** - A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers
- **[RePlug](https://arxiv.org/abs/2301.12652)** — Retriever‑aware generation
- **[RETRO](https://arxiv.org/abs/2112.04426)** — Retro‑fitted retrieval
- **[Streaming RAG](https://github.com/run-llama/cookbook/blob/main/streaming.md)** — Low latency


### Multimodal RAG

- **[Multimodal RAG with CLIP](https://github.com/run-llama/llama_index/tree/main/docs/examples/multi_modal)** - Text-Image retrieval using CLIP
- **[SAM-RAG](https://arxiv.org/abs/2410.11321)** - Self-adaptive multimodal RAG framework
- **[ColPali](https://arxiv.org/abs/2407.01449)** - Efficient document retrieval with vision language models
- **[Building Multimodal RAG Systems](https://blog.llamaindex.ai/multi-modal-rag-621de7525fea)**

### Graph-based RAG

- **[Microsoft GraphRAG](https://github.com/microsoft/graphrag)** - Knowledge graph approach to RAG Research: [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- **[Knowledge Graph Integration for RAG](https://github.com/NirDiamant/RAG_Techniques/tree/main/all_rag_techniques/graph_rag)**
- **[Neo4j GraphRAG](https://neo4j.com/developer/graph-data-science/applied-graph-ml/)** - Building knowledge graphs for RAG


## Retrieval Methods

### Dense Retrieval

- **[Dense Passage Retrieval (DPR) Implementation](https://github.com/facebookresearch/DPR)**
- **[ColBERTv2: Effective and Efficient Retrieval](https://arxiv.org/abs/2112.01488)**

### Sparse Retrieval

- **[SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)** - Neural sparse retrieval with term expansion
- **[HNSW vs DiskANN](https://cazton.com/blogs/technical/hnsw-vs-diskann/)** 

### Hybrid Search

- **[Hybrid Search: Combining Dense and Sparse Retrieval](https://docs.opensearch.org/docs/latest/vector-search/ai-search/hybrid-search/index/)** - Implementation guide for hybrid search systems
- **[Dense‑Sparse‑Dense (DSD)](https://arxiv.org/abs/2404.09842)**
- **[Advanced Reranking Techniques](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb)** - Guide to implementing cross-encoder reranking.

More here: [All RAG Reranking (GitHub)](https://github.com/NirDiamant/RAG_Techniques/tree/main/all_rag_techniques)


### Other Techniques

* **[RAG Fusion](https://arxiv.org/abs/2402.03367)**
* **[Sentence Window Retrieval](https://generativeai.pub/advanced-rag-retrieval-strategies-sentence-window-retrieval-b6964b6e56f7)**
* **[Cross‑Encoder Re‑Ranking](https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/applications/retrieve_rerank/README.md)**
* [Gemini Small‑to‑Big Retriever](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/small_to_big_rag/small_to_big_rag.ipynb)
* **[Multi‑Vector Retrieval](https://arxiv.org/abs/2312.06644)**
* [Negative PRF (Pseudo‑Relevance)](https://dl.acm.org/doi/10.1145/3570724)

## Prompting Strategies

- **[RAG Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/research/rag)** - Comprehensive guide to prompt engineering for RAG systems

- **[LangChain RAG Prompt Hub](https://smith.langchain.com/hub/langchain-ai/rag-prompt)** - Collection of tested RAG prompt templates

- **[Efficient Prompt Engineering for RAG](https://iamholumeedey007.medium.com/prompt-engineering-patterns-for-successful-rag-implementations-b2707103ab56)** - Strategies for optimizing prompts in RAG systems

- **[Secure RAG applications using prompt engineering on Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/secure-rag-applications-using-prompt-engineering-on-amazon-bedrock//)** - Best practices for RAG prompts with security considerations

* [Zero‑Shot / Few‑Shot](https://www.promptingguide.ai/techniques/zeroshot)
* [Chain‑of‑Thought (CoT)](https://www.promptingguide.ai/techniques/cot)
* [Meta Prompting](https://www.promptingguide.ai/techniques/meta-prompting)
* [Generated Knowledge Prompting](https://www.promptingguide.ai/techniques/knowledge)
* [ReAct](https://arxiv.org/abs/2210.03629)
* [Reflexion](https://www.promptingguide.ai/techniques/reflexion)
* [Automatic Prompt Engineer (APE)](https://www.promptingguide.ai/techniques/ape)
* [Directional Stimulus Prompting (DSP)](https://www.promptingguide.ai/techniques/dsp)
* [Chain‑of‑Verification (CoVe)](https://learnprompting.org/docs/advanced/self_criticism/chain_of_verification?srsltid=AfmBOoqz8BIeQV9TaNk_P7mWO_ov2QAdcTCHMPjSjS4ZqNSAgJl9vH6Y)
* [Self‑Consistency](https://www.promptingguide.ai/techniques/consistency)
* [Prompt Compression](https://towardsdatascience.com/how-to-cut-rag-costs-by-80-using-prompt-compression-877a07c6bedb)
* [Dynamic / Adaptive Prompts](https://learnprompting.org/docs/trainable/dynamic-prompting?srsltid=AfmBOoqDry9oY5AbUH7HyDQy9wdAA6SJfVEeOzcoDCrcwrJDaht43qPH)
* System → Retrieval → User triple‑prompt
* [GraphPrompt](https://arxiv.org/abs/2302.08043)
* [Emerging RAG & Prompt Engineering Architectures for LLMs](https://cobusgreyling.medium.com/updated-emerging-rag-prompt-engineering-architectures-for-llms-17ee62e5cbd9)
* [How to Cut RAG Costs by 80% Using Prompt Compression](https://towardsdatascience.com/how-to-cut-rag-costs-by-80-using-prompt-compression-877a07c6bedb)

## Chunking & Pre‑processing

- **[11 Chunking Strategies for RAG — Simplified & Visualized](https://masteringllm.medium.com/11-chunking-strategies-for-rag-simplified-visualized-df0dbec8e373)** - Comprehensive guide covering 11 chunking methods with visual comparisons
- **[5 Levels of Text Splitting](https://www.linkedin.com/posts/danieljbukowski_the-5-levels-of-text-splitting-for-retrieval-activity-7151917158640275456-jKE0/)** - Hierarchical approach to chunking from basic to advanced
- **[Semantic Chunking with LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/)** - Implementation guide for semantic-based document splitting
- **[Optimizing Retrieval-Augmented Generation with Advanced Chunking Techniques](https://antematter.io/blogs/optimizing-rag-advanced-chunking-techniques-study)** - Research on optimal chunk sizes for different use cases
- **[CharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.CharacterTextSplitter.html/)** — fixed‑size
- **[RecursiveTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html/)**
- **[SentenceSplitter](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/)** (LlamaIndex)
- **[Unstructured‑IO loaders](https://github.com/Unstructured-IO/unstructured)**
- **[LoRA Chunking](https://github.com/mesolitica/Chunk-loss-LoRA)** - Fused kernel chunk loss to include LoRA to reduce memory, support DeepSpeed ZeRO3
- **[Semantic chunking video](https://youtu.be/8OJC21T2SL4)**
- **[Agentic chunking demo](https://youtu.be/8OJC21T2SL4?t=2882)** - The 5 Levels Of Text Splitting For Retrieval
- **[Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)**
- **[Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)**
- **[How to Chunk Text Data — A Comparative Analysis](https://towardsdatascience.com/how-to-chunk-text-data-a-comparative-analysis-3858c4a0997a)**

### Comparison Guides

- **[Vector Database Comparison: Pinecone vs Weaviate vs Chroma](https://medium.com/@rohitupadhye799/comparing-chroma-db-weaviate-and-pinecone-which-vector-database-is-right-for-you-3b85b561b3a3)** - Comprehensive enterprise-focused comparison with performance metrics

- **[Top Vector Database for RAG: Qdrant vs Weaviate vs Pinecone](https://research.aimultiple.com/vector-database-for-rag/)** - Performance comparison of 6 vector databases for RAG workloads

## Embeddings Models

- **[Embedding Model Comparison: OpenAI vs Cohere vs Open Source](https://blog.timescale.com/blog/evaluating-open-source-vs-openai-embeddings-for-rag-a-how-to-guide/)** - Comprehensive evaluation of commercial and open-source embedding models

- **[Best Embedding Model  — OpenAI / Cohere / Google / E5 / BGE](https://medium.com/@lars.chr.wiik/best-embedding-model-openai-cohere-google-e5-bge-931bfa1962dc/)** - Detailed comparison of top embedding models with performance metrics

- **[Matryoshka Embeddings for RAG](https://huggingface.co/blog/matryoshka)** - Implementing variable-size embeddings for efficiency

- **[BGE M3 and SPLADE Implementation Guide](https://zilliz.com/learn/bge-m3-and-splade-two-machine-learning-models-for-generating-sparse-embeddings)** - Guide to implementing sparse and dense embeddings



## Instruction Tuning & Optimization

* **[RA‑DIT](https://openreview.net/forum?id=22OTbutug9)**
* **[InstructRetro](https://arxiv.org/abs/2310.07713)**
* **[FLARE / Active RAG](https://github.com/jzbjyb/FLARE)**
* **[UltraFeedback](https://arxiv.org/abs/2309.15140)** — RLHF on RAG
* **[DSI‑T](https://arxiv.org/abs/2212.14024)** — Decoder‑only retrieval

## Finetuning and Training
- [Fine-Tuning Llama 2.0 with Single GPU Magic](https://ai.plainenglish.io/fine-tuning-llama2-0-with-qloras-single-gpu-magic-1b6a6679d436)
- [Practitioners guide to fine-tune LLMs for domain-specific use case](https://cismography.medium.com/practitioners-guide-to-fine-tune-llms-for-domain-specific-use-case-part-1-4561714d874f)
- [Are You Pre-training your RAG Models on Your Raw Text?](https://medium.com/thirdai-blog/are-you-pre-training-your-rag-models-on-your-raw-text-40f832d87703)
- [Combine Multiple LoRA Adapters for Llama 2](https://towardsdatascience.com/combine-multiple-lora-adapters-for-llama-2-ea0bef9025cf)
- [RAG vs Finetuning — Which Is the Best Tool to Boost Your LLM Application?](https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7)

## Response Quality, and Hallucination

* **[RAGTruth: A Hallucination Corpus](https://arxiv.org/abs/2401.00396)** - Dataset with 18,000 RAG responses and hallucination annotations
* **[Reducing Hallucination in Structured Outputs via RAG](https://arxiv.org/abs/2404.08189)**
* **[WhyLabs AI Control Center](https://whylabs.ai/)** - Platform for real-time guardrails and monitoring
* **[Vectara Hallucination Score](https://github.com/vectara/hallucination-leaderboard/)**
* **[Prompt‑Injection Defense](https://hiddenlayer.com/innovation-hub/prompt-injection-attacks-on-llms/)**
* **[OpenAI Function Calling JSON Schema](https://platform.openai.com/docs/guides/function-calling)**
* **[Harmless RLHF pipelines](https://huggingface.co/blog/rlhf)**
* **[in-Of-Verification Reduces Hallucination in LLMs](https://cobusgreyling.medium.com/chain-of-verification-reduces-hallucination-in-llms-20af5ea67672)**
* **[How to Detect Hallucinations in LLMs](https://towardsdatascience.com/real-time-llm-hallucination-detection-9a68bb292698)**
* **[Measuring Hallucinations in RAG Systems](https://vectara.com/measuring-hallucinations-in-rag-systems/)**


## Security and Privacy Considerations

- **[OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)** - Comprehensive security framework covering RAG vulnerabilities
- **[CSA RAG Security Best Practices](https://cloudsecurityalliance.org/blog/2024/02/12/retrieval-augmented-generation-rag-security-best-practices/)** - Enterprise-grade security controls for RAG
- **[Microsoft Presidio for PII Protection](https://github.com/microsoft/presidio)** - Framework for detecting and anonymizing sensitive information
- **[LLM Guard](https://github.com/protectai/llm-guard)** - Security toolkit for protecting LLM applications
- **[Masking PII Data in RAG Pipeline](https://betterprogramming.pub/masking-pii-data-in-rag-pipeline-326d2d330336)**
- **[Hijacking Chatbots: Dangerous Methods Manipulating GPTs](https://medium.com/@jankammerath/hijacking-chatbots-dangerous-methods-manipulating-gpts-52342f4f88b8)**
- **[Guardrails AI](https://github.com/guardrails-ai/guardrails)** - Framework for implementing security guardrails
- **[NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)** - Comprehensive toolkit for building programmable guardrails
- **[NeMo Guardrails: The Missing Manual](https://www.pinecone.io/learn/nemo-guardrails-intro/)**
- **[Safeguarding LLMs with Guardrails](https://towardsdatascience.com/safeguarding-llms-with-guardrails-4f5d9f57cff2)**

## Evaluation Metrics and Benchmarks

- **[RAGAS (Retrieval-Augmented Generation Assessment)](https://github.com/explodinggradients/ragas)** - Reference-free evaluation framework with component-level metrics
- **[TruLens](https://github.com/truera/trulens)** - Comprehensive evaluation and tracking for LLM applications
- **[DeepEval](https://github.com/confident-ai/deepeval)** - Open-source evaluation framework for LLMs
- **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** - Open-source observability platform
- **[RAGBench](https://huggingface.co/datasets/rungalileo/ragbench)** - 100k examples across 5 industry domains
- **[BeIR](https://github.com/beir-cellar/beir)** - Benchmark for zero-shot evaluation of information retrieval
- **[MTEB](https://huggingface.co/spaces/mteb/leaderboard)** - Massive Text Embedding Benchmark
- **[ARES](https://github.com/stanford-futuredata/ares)** - Automated Evaluation of RAG Systems
- **[RGB Benchmark](https://github.com/chen700564/RGB)** - implementation for Benchmarking Large Language Models in Retrieval-Augmented Generation
- **[LlamaIndex RAG eval](https://docs.llamaindex.ai/en/stable/examples/cookbooks/oreilly_course_cookbooks/Module-3/Evaluating_RAG_Systems/)** - Evaluation and benchmarking are crucial in developing LLM applications

#### Blogs
- [RAG Evaluation](https://cobusgreyling.medium.com/rag-evaluation-9813a931b3d4)
- [Evaluating RAG: A journey through metrics](https://www.elastic.co/search-labs/blog/articles/evaluating-rag-metrics)
- [Exploring End-to-End Evaluation of RAG Pipelines](https://betterprogramming.pub/exploring-end-to-end-evaluation-of-rag-pipelines-e4c03221429)
- [Evaluation Driven Development, the Swiss Army Knife for RAG Pipelines](https://levelup.gitconnected.com/evaluation-driven-development-the-swiss-army-knife-for-rag-pipelines-dba24218d47e)
- [Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)

### RAG Benchmark 2023

* [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431)
* [RECALL: A Benchmark for LLMs Robustness against External Counterfactual Knowledge](https://arxiv.org/abs/2311.08147)
* [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476)
* [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)

### RAG Benchmark 2024

* [CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2401.17043v2)
* [FeB4RAG: Evaluating Federated Search in the Context of Retrieval Augmented Generation](https://arxiv.org/abs/2402.11891)
* [CodeRAG-Bench: Can Retrieval Augment Code Generation?](https://arxiv.org/abs/2406.14497)
* [Long<sup>2</sup>RAG: Evaluating Long-Context & Long-Form Retrieval-Augmented Generation with Key Point Recall](https://arxiv.org/abs/2410.23000)

## Advantages and Disadvantages

* **[Advantages overview](https://towardsdatascience.com/retrieval-augmented-generation-intuitively-and-exhaustively-explain-6a39d6fe6fc9)**
* **[Disadvantages & pitfalls](https://medium.com/@kelvin.lu.au/disadvantages-of-rag-5024692f2c53)**
* **[RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study](https://arxiv.org/abs/2401.08406)**


## Performance, Cost & Observability

* **[Vector Database Optimization](https://medium.com/intel-tech/optimize-vector-databases-enhance-rag-driven-generative-ai-90c10416cb9c)** - Techniques for efficient vector storage and retrieval
* **[Hybrid Retrieval Strategies](https://arxiv.org/html/2503.23013v1#:~:text=Hybrid%20retrieval%20techniques%20in%20Retrieval,to%20adjust%20to%20different%20queries.)** - Combining multiple retrieval methods for better performance
* **[Chunking Optimization](https://medium.com/@ayoubkirouane3/simple-chunking-strategies-for-rag-applications-part-1-d56903b167c5)** - Strategies for optimal text segmentation
* **[LangFuse](https://github.com/langfuse/langfuse)**
* **[LangSmith](https://docs.smith.langchain.com)**
* **[Helicone](https://github.com/Helicone/helicone)** — telemetry
* **[WandB RAG guide](https://wandb.ai/wandb-japan/rag-hands-on/reports/Step-for-developing-and-evaluating-RAG-application-with-W-B--Vmlldzo1NzU4OTAx)**
* **[OpenLLMetry](https://github.com/traceloop/openllmetry)** - Open-source observability for your LLM application, based on OpenTelemetry
* **[Cost optimisation tips](https://medium.com/madhukarkumar/secrets-to-optimizing-rag-llm-apps-for-better-accuracy-performance-and-lower-cost-da1014127c0a)**



### Cost Calculators
- **[RAG Cost Calculator](https://www.rag.mjacques.co/start)** - Tool for estimating and optimizing RAG pipeline costs
- [RAG Savings Calculator](https://www.vectara.com/business/resources/rag-savings-calculator)
- [RAG Cost Calculator](https://zilliz.com/rag-cost-calculator/)

## RAG Fine-tuning

- **[RAFT (Retrieval Augmented Fine-Tuning)](https://arxiv.org/abs/2403.10131)** - Adapting Language Model to Domain Specific RAG
- **[Fine-tuning vs RAG Guide](https://www.cohere.com/blog/rag-vs-fine-tuning)** - Comprehensive comparison and guidance
- **[Direct Preference Optimization (DPO) for RAG](https://arxiv.org/abs/2305.18290)** - Alternative to RLHF for aligning RAG outputs

## Knowledge‑Graph / Structured RAG

* **[DBpedia](https://dbpedia.org/)** - Structured knowledge from Wikipedia
* **[Wikidata](https://www.wikidata.org/)** - Community-maintained knowledge base
* **[ConceptNet](https://conceptnet.io/)** - Large-scale commonsense knowledge graph
* **[YAGO](https://yago-knowledge.org/)** - High-quality knowledge base
* **[Neo4j LLM Knowledge Graph Builder](https://github.com/neo4j-labs/llm-graph-builder)**
* **[Neo4j RAG blog](https://neo4j.com/blog/rag//)**
* **[GraphRAG site](https://graphrag.com)**
* **[NebulaGraph Graph‑RAG article](https://medium.com/@nebulagraph/graph-rag-the-new-llm-stack-with-knowledge-graphs-e1e902c504ed)**

## Libraries and SDKs

- **[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)** - Python framework for sentence, text and image embeddings
- **[LiteLLM](https://github.com/BerriAI/litellm)** - Python SDK for 100+ LLM APIs in OpenAI format
- **[AI SDK](https://github.com/vercel/ai)** - TypeScript toolkit for building AI applications
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - State-of-the-art ML for PyTorch, TensorFlow, and JAX

## Key Concepts
- **[Hugging Face Transformers - RAG Documentation](https://huggingface.co/docs/transformers/model_doc/rag)**
- **[RAG-Survey GitHub Repository](https://github.com/hymie122/RAG-Survey)** - Curated collection of RAG papers with taxonomy


## Educational Content

### Courses and Tutorials

* **[Building Advanced RAG (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)**
* **[RAG from Scratch (FreeCodeCamp)](https://www.freecodecamp.org/news/tag/rag//)**
* **[IBM Generative AI and RAG Course (Coursera)](https://www.coursera.org/learn/generative-ai-llm-architecture-data-preparation)**
* [**MAGMaR 2024** — Multimodal Augmented Generation (NeurIPS)](https://marworkshop.github.io/neurips24/)
* **[ACL 2024 Knowledgeable LMs Tutorial](https://aclanthology.org/2024.acl-tutorials.3)**
* **[SIGIR 2023 Generative IR Workshop](https://arxiv.org/abs/2306.02887)**

- [**MAGMaR - The 1st Workshop on Multimodal Augmented Generation via MultimodAl Retrieval** - *Reno Kriz, Kenton Murray, Eugene Yang, Francis Ferraro, Kate Sanders, Cameron Carpenter, Benjamin Van Durme*](https://nlp.jhu.edu/magmar)
- [**Towards Knowledgeable Language Models** - *Zoey Sha Li, Manling Li, Michael JQ Zhang, Eunsol Choi, Mor Geva, Peter Hase*](https://knowledgeable-lm.github.io/)

- **Modular RAG and RAG Flow**  *Yunfan Gao* (2024) Tutorial - [Blog I](https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-%E2%85%B0-e69b32dc13a3) and [Blog II](https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-ii-77b62bf8a5d3)

- **Stanford CS25: V3 I Retrieval Augmented Language Models**  *Douwe Kiela* (2023) Lecture - [Video](https://www.youtube.com/watch?v=mE7IDf2SmJg&ab_channel=StanfordOnline)

- **Building RAG-based LLM Applications for Production**  *Anyscale* (2023) Tutorial - [Blog](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

- **Multi-Vector Retriever for RAG on tables, text, and images**  *LangChain* (2023) Tutorial - [Blog](https://blog.langchain.dev/semi-structured-multi-modal-rag)

- **Retrieval-based Language Models and Applications**  *Asai et al.* (2023) Tutorial  ACL [Website](https://acl2023-retrieval-lm.github.io/) and [Video](https://us06web.zoom.us/rec/play/6fqU9YDLoFtWqpk8w8I7oFrszHKW6JkbPVGgHsdPBxa69ecgCxbmfP33asLU3DJ74q5BXqDGR2ycOTFk.93teqylfi_uiViNK?canPlayFromShare=true&from=share_recording_detail&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Fus06web.zoom.us%2Frec%2Fshare%2FNrYheXPtE5zOlbogmdBg653RIu7RBO1uAsYH2CZt_hacD1jOHksRahGlERHc_Ybs.KGX1cRVtJBQtJf0o)

- **Advanced RAG Techniques: an Illustrated Overview**  *Ivan Ilin* (2023) Tutorial  - [Blog](https://towardsai.net/p/machine-learning/advanced-rag-techniques-an-illustrated-overview)

- **Retrieval Augmented Language Modeling**  *Melissa Dell* (2023) Lecture  [Video](https://www.youtube.com/watch?v=XC4eFiIMOmY)


* **[RAG from scratch](https://github.com/langchain-ai/rag-from-scratch)**
* **[Building RAG Applications for Production](https://github.com/ray-project/llm-applications)**
* **[Stanford CS25: V3 I Retrieval Augmented Language Models](https://www.youtube.com/watch?v=mE7IDf2SmJg)** 
* **[Ray Summit 2024: Production RAG Pipelines](https://www.youtube.com/playlist?list=PLzTswPQNepXlEfa0KHBlyfqgZ0WBrJ52Z)**
* **[Haystack RAG Workshop](https://www.youtube.com/watch?v=8qqaqefugWQ)**
* **[Azure Cognitive Search RAG Tutorial](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview?tabs=docs)**

### Blogs and Articles

* **[RAG Implementation with LangChain and Weaviate](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)** - From theory to Python implementation
* **[Advanced RAG Techniques: An Illustrated Overview](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)**
* **[The RAGOps Stack: Critical Components](https://towardsdatascience.com/ragops-guide-building-and-scaling-retrieval-augmented-generation-systems-3d26b3ebd627/)**
* **[Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/)**
* **[RAG Intuitively & Exhaustively Explained](https://towardsdatascience.com/retrieval-augmented-generation-intuitively-and-exhaustively-explain-6a39d6fe6fc9)**
* **[RAG in Production: 9 Lessons](https://medium.com/@jalajagr/9-hard-earned-lessons-for-rag-in-prod-ead56abaed52/)**
* **[Reranking vs Embeddings on Cursor](https://medium.com/@gonzalo.mordecki/reranking-vs-embeddings-on-cursor-a2d728ba67dd)**
* **[Forget RAG, Think RAG‑Fusion](https://medium.com/data-science/forget-rag-the-future-is-rag-fusion-1147298d8ad1)**
* **[Hidden Costs of RAG](https://medium.com/@kelvin.lu.au/disadvantages-of-rag-5024692f2c53)**


### Newsletters & Forums
* [ragaboutit](https://ragaboutit.com/) - A blog and newsletter focused specifically on RAG news, tutorials, and insights, making it a dedicated resource for staying up-to-date.
* [r/LangChain](https://www.reddit.com/r/LangChain/) 
* [r/rag](https://www.reddit.com/r/Rag/) - Reddit communities for practical discussions, troubleshooting, and sharing projects. These are valuable for seeing what challenges other developers are facing in real-time.


### Talks and Conferences

- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020)](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)**
- **[Self-RAG: Learning to Retrieve, Generate, and Critique (ICLR 2024)](https://arxiv.org/abs/2310.11511)**
- **[RAG Research Papers Collection](https://github.com/jxzhangjhu/Awesome-LLM-RAG)** - Curated list from ICML, ICLR, ACL

## Influential Researchers and Influencers

* **[Patrick Lewis](https://www.patricklewis.io/)** - Lead author of original RAG paper, AI Research Scientist at Cohere
* **[Sebastian Riedel](https://www.linkedin.com/in/riedel/?originalSubdomain=uk)** - Co-author of RAG paper, Professor at UCL and DeepMind
* **[Douwe Kiela](https://douwekiela.github.io/)** - Co-author of RAG paper, CEO of Contextual AI
* **[Gautier Izacard](https://scholar.google.com/citations?user=aL3MllMAAAAJ&hl=en)** - Author of FiD and Atlas papers, Meta AI
* **[Kelvin Guu](https://www.linkedin.com/in/kelvinguu/)** - Lead author of REALM paper, Google Research
* **[Douwe Kiela](https://www.linkedin.com/in/douwekiela/)** — Modular RAG, Stanford
* **[Matei Zaharia](https://www.linkedin.com/in/mateizaharia/)** — DSPy, Databricks
* **[Akari Asai](https://akariasai.github.io/)** — Dense retrieval research
* **[Jerry Liu](https://github.com/jerryjliu)** — LlamaIndex
* **[Harrison Chase](https://github.com/hwchase17)** — LangChain
* **[Andrej Karpathy](https://twitter.com/karpathy)** — LLM systems
* **[Jeff Dean](https://www.linkedin.com/in/jeff-dean-8b212555/)** — Google Research
* **[Artem Yankov](https://github.com/yankov)** — Qdrant
* **[Alden Do Rosario](https://www.linkedin.com/in/aldendorosario/)** - RAG Influencer, CEO `CustomGPT.ai`  


## Latest Trends 2024-2025

- RAG-as-a-Service market at $1.2B (2024)
- Projected 49.1% CAGR through 2030
- On-device RAG for privacy

## Community Resources

### Reddit
- **[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)** - 493k members
- **[r/MachineLearning](https://www.reddit.com/r/MachineLearning/)** - Active RAG discussions
- **[r/RAG](https://www.reddit.com/r/rag/)** - Dedicated RAG subreddit

### Discord
- **RAG TAGG Discord** - 2,492 members
- **[Vectara RAGTime Bot](https://www.vectara.com/blog/ragtime-a-rag-powered-bot-for-slack-and-discord)**
- **RAGHub Community**

### GitHub Communities
- **[RAGHub Repository](https://github.com/Andrew-Jang/RAGHub)**
- **[Microsoft GraphRAG](https://github.com/microsoft/graphrag)**



## Existing Collections

* **[Awesome-RAG (awesome-rag)](https://github.com/awesome-rag/awesome-rag)**
* **[RAG Resources (mrdbourke)](https://github.com/mrdbourke/rag-resources)**
* **[RAG Techniques (NirDiamant)](https://github.com/NirDiamant/RAG_Techniques)**
* **[Danielskry / Awesome‑RAG](https://github.com/Danielskry/Awesome-RAG)**
* **[frutik / Awesome‑RAG](https://github.com/frutik/Awesome-RAG)**
* **[coree / awesome‑rag](https://github.com/coree/awesome-rag)**
* **[jxzhangjhu / Awesome‑LLM‑RAG](https://github.com/jxzhangjhu/Awesome-LLM-RAG)**
* **[SJTU‑DMTai / awesome‑rag](https://github.com/SJTU-DMTai/awesome-rag)**
* **[lucifertrj / Awesome‑RAG](https://github.com/lucifertrj/Awesome-RAG)**



## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting a pull request.

## License

This collection is licensed under MIT.

