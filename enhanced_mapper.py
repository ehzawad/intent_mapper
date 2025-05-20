from dotenv import load_dotenv
import yaml
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
import time
from datetime import datetime

# Load environment variables
load_dotenv()
client = OpenAI()

class EmbeddingCache:
    def __init__(self, cache_file: str = "embedding_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, List[float]] = self._load_cache()

    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cache from file if it exists."""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from cache or compute and cache it."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash not in self.cache:
            embedding = self._compute_embedding(text)
            self.cache[text_hash] = embedding
            self._save_cache()
        
        return self.cache[text_hash]

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding using OpenAI API."""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

class SimpleVectorStore:
    def __init__(self):
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self.last_updated: Optional[float] = None

    def add_vector(self, text: str, embedding: List[float], metadata: Dict[str, Any]):
        """Add a vector to the store."""
        vector_id = hashlib.md5(text.encode()).hexdigest()
        self.vectors[vector_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }
        self.last_updated = time.time()

    def find_most_similar(self, query_embedding: List[float], threshold: float = 0.75) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """Find the most similar vector above the threshold."""
        best_score = 0
        best_match = None

        for vector_id, vector_data in self.vectors.items():
            similarity = self._calculate_similarity(query_embedding, vector_data["embedding"])
            if similarity > best_score:
                best_score = similarity
                best_match = (vector_id, vector_data)

        if best_match and best_score >= threshold:
            return best_match[1]["text"], best_match[1]["metadata"], best_score
        return None

    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        return sum(a * b for a, b in zip(emb1, emb2))

    def clear(self):
        """Clear all vectors."""
        self.vectors.clear()
        self.last_updated = None

class IntentMapper:
    def __init__(self, intents_file: str, threshold: float = 0.75):
        self.intents_file = intents_file
        self.threshold = threshold
        self.cache = EmbeddingCache()
        self.vector_store = SimpleVectorStore()
        self.last_file_check = 0
        self._initialize_vectors()

    def _initialize_vectors(self):
        """Initialize vector store with all questions from intents."""
        self.vector_store.clear()
        intents = self._load_intents()

        for intent_name, intent_data in intents['intents'].items():
            # Add canonical question
            canonical = intent_data['canonical_question']
            embedding = self.cache.get_embedding(canonical)
            self.vector_store.add_vector(
                canonical,
                embedding,
                {
                    "intent": intent_name,
                    "is_canonical": True,
                    "canonical_question": canonical,
                    "subcategory": None  # Parent-level intent
                }
            )

            # Add variations
            for variation in intent_data['variations']:
                embedding = self.cache.get_embedding(variation)
                self.vector_store.add_vector(
                    variation,
                    embedding,
                    {
                        "intent": intent_name,
                        "is_canonical": False,
                        "canonical_question": canonical,
                        "subcategory": None  # Parent-level intent
                    }
                )
            
            # Process subcategories if they exist
            if 'subcategories' in intent_data:
                for subcategory_name, subcategory_data in intent_data['subcategories'].items():
                    # Add subcategory canonical question
                    sub_canonical = subcategory_data['canonical_question']
                    embedding = self.cache.get_embedding(sub_canonical)
                    self.vector_store.add_vector(
                        sub_canonical,
                        embedding,
                        {
                            "intent": intent_name,
                            "subcategory": subcategory_name,
                            "is_canonical": True,
                            "canonical_question": sub_canonical
                        }
                    )

                    # Add subcategory variations
                    for sub_variation in subcategory_data['variations']:
                        embedding = self.cache.get_embedding(sub_variation)
                        self.vector_store.add_vector(
                            sub_variation,
                            embedding,
                            {
                                "intent": intent_name,
                                "subcategory": subcategory_name,
                                "is_canonical": False,
                                "canonical_question": sub_canonical
                            }
                        )

    def _load_intents(self) -> Dict:
        """Load intents from YAML file."""
        with open(self.intents_file, 'r') as file:
            return yaml.safe_load(file)

    def _check_file_changes(self):
        """Check if intents file has changed and reload if necessary."""
        current_time = time.time()
        if current_time - self.last_file_check > 5:  # Check every 5 seconds
            self.last_file_check = current_time
            file_mtime = Path(self.intents_file).stat().st_mtime
            if not self.vector_store.last_updated or file_mtime > self.vector_store.last_updated:
                print("Intents file changed, reloading vectors...")
                self._initialize_vectors()

    def find_intent(self, query: str) -> Dict[str, Any]:
        """Find the matching intent for a query."""
        self._check_file_changes()

        # Get query embedding
        query_embedding = self.cache.get_embedding(query)

        # Find most similar vector
        match = self.vector_store.find_most_similar(query_embedding, self.threshold)

        if match:
            text, metadata, confidence = match
            return {
                "intent": metadata["intent"],
                "subcategory": metadata.get("subcategory"), # Use .get() for safety, defaulting to None
                "canonical_question": metadata["canonical_question"],
                "confidence": confidence,
                "status": "success"
            }
        else:
            return {
                "status": "fallback",
                "confidence": 0,
                "message": "No intent matched with sufficient confidence"
            }

def main():
    # Initialize mapper
    mapper = IntentMapper('intents.yaml')

    # Test questions
    test_questions = [
        "How much do I have in my account?", # Existing: account_balance
        "Is my card working properly?", # Existing: card_status
        "What's my current balance?", # Existing: account_balance variation
        "Can you check my card status?", # Existing: card_status variation
        "I'm having an issue with a payment.", # New: payment_issues (parent)
        "My transaction was declined.", # New: payment_issues -> failed_payment
        "I don't recognize a charge.", # New: payment_issues -> unauthorized_payment
        "Something went wrong with my payment.", # New: payment_issues (parent variation)
        "Why did my payment not go through?", # New: payment_issues -> failed_payment variation
        "There's an unauthorized transaction on my account.", # New: payment_issues -> unauthorized_payment variation
        "Tell me about the weather",  # Should trigger fallback
    ]

    print("Testing question matching:")
    for question in test_questions:
        print(f"\nUser asks: {question}")
        result = mapper.find_intent(question)
        
        if result["status"] == "success":
            print(f"Matched intent: {result['intent']}")
            if result.get("subcategory"):
                print(f"Matched subcategory: {result['subcategory']}")
            print(f"Canonical question: {result['canonical_question']}")
            print(f"Confidence: {result['confidence']:.2f}")
        else:
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")

if __name__ == "__main__":
    main()
