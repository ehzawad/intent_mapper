# from dotenv import load_dotenv
# import yaml
# import json
# import time
# from pathlib import Path
# from typing import Dict, List, Any
# from openai import OpenAI
# import numpy as np
#
# # Load environment variables
# load_dotenv()
# client = OpenAI()
#
# class EmbeddingTrainer:
#     def __init__(self, intents_file: str, vector_store_file: str = "vector_store.json"):
#         self.intents_file = intents_file
#         self.vector_store_file = vector_store_file
#         self.vectors: Dict[str, Dict[str, Any]] = {}
#     
#     def _load_intents(self) -> Dict:
#         """Load intents from YAML file."""
#         with open(self.intents_file, 'r') as file:
#             return yaml.safe_load(file)
#     
#     def _get_embedding(self, text: str) -> List[float]:
#         """Get embedding for a text using OpenAI API."""
#         try:
#             response = client.embeddings.create(
#                 model="text-embedding-ada-002",
#                 input=text
#             )
#             return response.data[0].embedding
#         except Exception as e:
#             print(f"Error getting embedding for text: {text}")
#             print(f"Error: {str(e)}")
#             return []
#
#     def train(self):
#         """Process all intents and save their embeddings."""
#         print("Starting embedding training process...")
#         intents = self._load_intents()
#         
#         for intent_name, intent_data in intents['intents'].items():
#             print(f"\nProcessing intent: {intent_name}")
#             
#             # Process canonical question
#             canonical = intent_data['canonical_question']
#             print(f"Getting embedding for canonical question: {canonical}")
#             embedding = self._get_embedding(canonical)
#             
#             if embedding:
#                 self.vectors[canonical] = {
#                     "text": canonical,
#                     "embedding": embedding,
#                     "metadata": {
#                         "intent": intent_name,
#                         "is_canonical": True,
#                         "canonical_question": canonical
#                     }
#                 }
#             
#             # Process variations
#             print(f"Processing {len(intent_data['variations'])} variations...")
#             for variation in intent_data['variations']:
#                 print(f"Getting embedding for variation: {variation}")
#                 embedding = self._get_embedding(variation)
#                 
#                 if embedding:
#                     self.vectors[variation] = {
#                         "text": variation,
#                         "embedding": embedding,
#                         "metadata": {
#                             "intent": intent_name,
#                             "is_canonical": False,
#                             "canonical_question": canonical
#                         }
#                     }
#             
#             # Small delay to avoid rate limits
#             time.sleep(0.5)
#         
#         self._save_vectors()
#         print("\nTraining complete! Vector store has been updated.")
#     
#     def _save_vectors(self):
#         """Save vectors to file."""
#         print(f"\nSaving vectors to {self.vector_store_file}")
#         with open(self.vector_store_file, 'w') as f:
#             json.dump({
#                 "vectors": self.vectors,
#                 "last_updated": time.time()
#             }, f, indent=2)
#
# def main():
#     # Check if intents file exists
#     intents_file = "intents.yaml"
#     if not Path(intents_file).exists():
#         print(f"Error: {intents_file} not found!")
#         return
#     
#     # Initialize trainer
#     trainer = EmbeddingTrainer(intents_file)
#     
#     # Train embeddings
#     print(f"Starting training process using intents from {intents_file}")
#     trainer.train()
#
# if __name__ == "__main__":
#     main()


from dotenv import load_dotenv
import yaml
import time
from pathlib import Path
import chromadb
from openai import OpenAI
from typing import Dict, Any

# Load environment variables
load_dotenv()
client = OpenAI()

class EmbeddingTrainer:
    def __init__(self, intents_file: str, collection_name: str = "intent_embeddings"):
        self.intents_file = intents_file
        self.collection_name = collection_name
        
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
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
        print(f"Created new collection: {collection_name}")
    
    def _load_intents(self) -> Dict:
        """Load intents from YAML file."""
        with open(self.intents_file, 'r') as file:
            return yaml.safe_load(file)
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a text using OpenAI API."""
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding for text: {text}")
            print(f"Error: {str(e)}")
            return []

    def train(self):
        """Process all intents and store their embeddings in ChromaDB."""
        print("Starting embedding training process...")
        intents = self._load_intents()
        
        # Keep track of items to add
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for intent_name, intent_data in intents['intents'].items():
            print(f"\nProcessing intent: {intent_name}")
            
            # Process canonical question
            canonical = intent_data['canonical_question']
            print(f"Getting embedding for canonical question: {canonical}")
            canonical_embedding = self._get_embedding(canonical)
            
            if canonical_embedding:
                ids.append(f"{intent_name}_canonical")
                embeddings.append(canonical_embedding)
                metadatas.append({
                    "intent": intent_name,
                    "is_canonical": True,
                    "canonical_question": canonical
                })
                documents.append(canonical)
            
            # Process variations
            print(f"Processing {len(intent_data['variations'])} variations...")
            for i, variation in enumerate(intent_data['variations']):
                print(f"Getting embedding for variation: {variation}")
                embedding = self._get_embedding(variation)
                
                if embedding:
                    ids.append(f"{intent_name}_var_{i}")
                    embeddings.append(embedding)
                    metadatas.append({
                        "intent": intent_name,
                        "is_canonical": False,
                        "canonical_question": canonical
                    })
                    documents.append(variation)
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
        
        # Add all embeddings in one batch
        print("\nAdding embeddings to ChromaDB...")
        if ids:  # Only add if we have embeddings
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            print(f"\nTraining complete! Added {len(ids)} embeddings to ChromaDB.")
        else:
            print("\nNo embeddings to add. Training failed.")

def main():
    # Check if intents file exists
    intents_file = "intents.yaml"
    if not Path(intents_file).exists():
        print(f"Error: {intents_file} not found!")
        return
    
    # Initialize trainer
    trainer = EmbeddingTrainer(intents_file)
    
    # Train embeddings
    print(f"Starting training process using intents from {intents_file}")
    trainer.train()

if __name__ == "__main__":
    main()
