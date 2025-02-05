# Intent Mapper

A fast and efficient intent mapping system using OpenAI embeddings and ChromaDB for semantic similarity matching. The system can understand user queries and map them to predefined intents with high accuracy, falling back to GPT-4 for unmatched queries.


## Features

- Fast semantic similarity matching using ChromaDB
- Embedding caching for improved performance
- Batch processing for training
- Async support for quick responses
- Fallback to GPT-4 for unmatched queries
- Progress tracking and logging
- Configurable similarity thresholds

## Booting up

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file from `.env.example` file

## Basic Usage

1. Define your intents in `intents.yaml`:
```yaml
intents:
  account_balance:
    canonical_question: "What is your account balance?"
    variations:
      - "Can you tell me how much money I've got in my account?"
      - "Could you check my current balance?"
      
  card_status:
    canonical_question: "What is your card status?"
    variations:
      - "Is my card active?"
      - "Can you check if my card is working?"
```

2. Train the embeddings:
```bash
python train_embeddings.py
```

3. Test queries:
```bash
python fast_matcher.py
```

## Development Progression

### Phase 1: Basic Implementation ✅
- Simple intent matching
- OpenAI embeddings
- Basic similarity search

### Phase 2: Performance Optimization ✅
- ChromaDB integration
- Embedding caching
- Batch processing

### Phase 3: Enhanced Features ✅
- Async support
- GPT-4 fallback
- Response time tracking

### Phase 4: Production Readiness 🚧
- Comprehensive testing
- Error handling
- Monitoring
- Documentation

### Phase 5: Future Improvements 🎯
- API interface
- More sophisticated matching
- Multi-language support
- Custom embeddings


A break-down each component:

1. Data Input Layer
   - `intents.yaml`: Source of truth for intents and variations
   - `train_embeddings.py`: Base training script
   - `train_embeddings_batch.py`: Optimized batch training

2. External APIs (OpenAI)
   - Embeddings API: Converts text to vectors
   - GPT-4 Assistant API: Handles fallback responses

3. Storage Systems
   - ChromaDB: Vector database for embeddings
   - Embedding Cache: TTL cache for frequent queries

4. Core Processing
   - `fast_matcher.py`: Main processing engine
   - Similarity Matching: Using cosine similarity
   - Intent Resolution: Maps to canonical questions

5. Testing & Validation
   - `test_question.py`: Basic testing interface
   - `test_question_parallel.py`: Parallel testing
   - `basic_openai_test.py`: API sanity checks

Key Data Flows:
1. Training Flow:
   ```
   YAML -> Training Script -> Embeddings API -> ChromaDB
   ```

2. Query Flow:
   ```
   Query -> Cache Check -> Embedding -> Similarity -> Intent/Fallback
   ```
