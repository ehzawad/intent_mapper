from dotenv import load_dotenv
import yaml
import time
from pathlib import Path
import chromadb
from openai import OpenAI
from typing import Dict, List, Any
from tqdm import tqdm  # For progress bars

# Load environment variables
load_dotenv()
client = OpenAI()

class EmbeddingTrainer:
    def __init__(self, intents_file: str, collection_name: str = "intent_embeddings", batch_size: int = 20):
        self.intents_file = intents_file
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Delete collection if exists and create new
        try:
            self.chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            print(f"No existing collection found: {collection_name}")
            
        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {collection_name}")
    
    def _load_intents(self) -> Dict:
        """Load intents from YAML file."""
        with open(self.intents_file, 'r') as file:
            return yaml.safe_load(file)
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error getting embeddings batch: {str(e)}")
            return []

    def _prepare_batches(self, items: List[tuple]) -> List[List[tuple]]:
        """Split items into batches."""
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]

    def train(self):
        """Process all intents and store their embeddings in ChromaDB using batches."""
        print("Starting embedding training process...")
        intents = self._load_intents()
        
        # Prepare all items for batch processing
        all_items = []
        
        # Collect all items to process
        for intent_name, intent_data in intents['intents'].items():
            # Add canonical question
            canonical = intent_data['canonical_question']
            all_items.append((
                f"{intent_name}_canonical",
                canonical,
                {
                    "intent": intent_name,
                    "is_canonical": True,
                    "canonical_question": canonical
                }
            ))
            
            # Add variations
            for i, variation in enumerate(intent_data['variations']):
                all_items.append((
                    f"{intent_name}_var_{i}",
                    variation,
                    {
                        "intent": intent_name,
                        "is_canonical": False,
                        "canonical_question": canonical
                    }
                ))
        
        # Process in batches
        total_batches = (len(all_items) + self.batch_size - 1) // self.batch_size
        print(f"\nProcessing {len(all_items)} items in {total_batches} batches...")
        
        for batch in tqdm(self._prepare_batches(all_items), total=total_batches, desc="Processing batches"):
            ids = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            metadatas = [item[2] for item in batch]
            
            # Get embeddings for batch
            embeddings = self._get_embeddings_batch(texts)
            
            if embeddings:
                # Add batch to ChromaDB
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        print(f"\nTraining complete! Added {len(all_items)} embeddings to ChromaDB.")
        print(f"Collection info: {self.collection.count()} items stored")

def main():
    # Check if intents file exists
    intents_file = "intents.yaml"
    if not Path(intents_file).exists():
        print(f"Error: {intents_file} not found!")
        return
    
    # Initialize trainer with batch size of 20
    trainer = EmbeddingTrainer(intents_file, batch_size=20)
    
    # Train embeddings
    print(f"Starting training process using intents from {intents_file}")
    trainer.train()

if __name__ == "__main__":
    main()
