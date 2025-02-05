# from dotenv import load_dotenv
# import json
# import time
# from openai import OpenAI
# import numpy as np
# from typing import Dict, List, Any
#
# load_dotenv()
# client = OpenAI()
#
# class QuestionMatcher:
#     def __init__(self, vector_store_file: str = "vector_store.json", threshold: float = 0.85):
#         self.vector_store_file = vector_store_file
#         self.threshold = threshold
#         self.vectors = self._load_vectors()
#         self.assistant = self._create_assistant()
#     
#     def _create_assistant(self):
#         """Create a GPT-4-0 assistant for fallback responses."""
#         return client.beta.assistants.create(
#             name="Fallback Handler",
#             model="gpt-4-1106-preview",
#             instructions="""You are a helpful assistant that provides very concise responses.
#             Always respond in 1-2 sentences without any formatting, bullets, or quotes.
#             Be direct and informative while maintaining a friendly tone."""
#         )
#     
#     def _load_vectors(self) -> Dict:
#         """Load vectors from file."""
#         with open(self.vector_store_file, 'r') as f:
#             data = json.load(f)
#             return data["vectors"]
#     
#     def _get_embedding(self, text: str) -> List[float]:
#         """Get embedding for a text using OpenAI API."""
#         response = client.embeddings.create(
#             model="text-embedding-ada-002",
#             input=text
#         )
#         return response.data[0].embedding
#     
#     def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
#         """Calculate cosine similarity between two embeddings."""
#         vec1 = np.array(emb1)
#         vec2 = np.array(emb2)
#         return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#     
#     def _get_fallback_response(self, question: str) -> str:
#         """Get a fallback response using GPT-4-0 assistant."""
#         try:
#             # Create a thread
#             thread = client.beta.threads.create()
#             
#             # Add the user's question
#             client.beta.threads.messages.create(
#                 thread_id=thread.id,
#                 role="user",
#                 content=question
#             )
#             
#             # Run the assistant
#             run = client.beta.threads.runs.create(
#                 thread_id=thread.id,
#                 assistant_id=self.assistant.id
#             )
#             
#             # Wait for completion
#             while True:
#                 run_status = client.beta.threads.runs.retrieve(
#                     thread_id=thread.id,
#                     run_id=run.id
#                 )
#                 if run_status.status == 'completed':
#                     break
#                 elif run_status.status in ['failed', 'cancelled', 'expired']:
#                     return "I apologize, but I'm not able to help with that right now."
#                 time.sleep(0.5)
#             
#             # Get the assistant's response
#             messages = client.beta.threads.messages.list(
#                 thread_id=thread.id
#             )
#             
#             for message in messages.data:
#                 if message.role == "assistant":
#                     return message.content[0].text.value
#             
#             return "I apologize, but I'm not able to help with that right now."
#             
#         except Exception as e:
#             return "I apologize, but I'm not able to help with that right now."
#     
#     def match_question(self, question: str) -> Dict[str, Any]:
#         """Find the best matching intent for a question."""
#         # Get embedding for the question
#         query_embedding = self._get_embedding(question)
#         
#         # Find best match
#         best_score = 0
#         best_match = None
#         
#         for _, vector_data in self.vectors.items():
#             similarity = self._calculate_similarity(query_embedding, vector_data["embedding"])
#             if similarity > best_score:
#                 best_score = similarity
#                 best_match = vector_data
#         
#         # Check if match meets threshold
#         if best_match and best_score >= self.threshold:
#             return {
#                 "status": "success",
#                 "intent": best_match["metadata"]["intent"],
#                 "canonical_question": best_match["metadata"]["canonical_question"],
#                 "confidence": best_score
#             }
#         
#         # Get fallback response
#         fallback_response = self._get_fallback_response(question)
#         return {
#             "status": "nlu_fallback",
#             "confidence": best_score,
#             "response": fallback_response
#         }
#
#     def cleanup(self):
#         """Cleanup resources."""
#         try:
#             client.beta.assistants.delete(self.assistant.id)
#         except Exception:
#             pass
#
# def main():
#     # Initialize matcher
#     matcher = QuestionMatcher()
#     
#     try:
#         print("\nQuestion Matching System")
#         print("Enter your question (or 'quit' to exit)")
#         
#         while True:
#             question = input("\nYour question: ").strip()
#             
#             if question.lower() in ['quit', 'exit', 'q']:
#                 break
#             
#             if not question:
#                 continue
#             
#             result = matcher.match_question(question)
#             
#             if result["status"] == "success":
#                 print(f"Matched intent: {result['intent']}")
#                 print(f"Canonical question: {result['canonical_question']}")
#                 print(f"Confidence: {result['confidence']:.2f}")
#             else:
#                 print(f"Status: {result['status']}")
#                 print(f"Confidence: {result['confidence']:.2f}")
#                 print(f"Response: {result['response']}")
#     
#     finally:
#         # Cleanup
#         matcher.cleanup()
#
# if __name__ == "__main__":
#     main()
#
#
#
from dotenv import load_dotenv
import time
import chromadb
from openai import OpenAI
from typing import Dict, Any

# Load environment variables
load_dotenv()
client = OpenAI()

class QuestionMatcher:
    def __init__(self, collection_name: str = "intent_embeddings", threshold: float = 0.85):
        self.threshold = threshold
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get the collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"Connected to collection: {collection_name}")
        except Exception as e:
            print(f"Error: Collection not found. Please run training first.")
            raise e
            
        # Initialize assistant for fallback
        self.assistant = self._create_assistant()
    
    def _create_assistant(self):
        """Create a GPT-4 assistant for fallback responses."""
        return client.beta.assistants.create(
            name="Fallback Handler",
            model="gpt-4-1106-preview",
            instructions="""You are a helpful assistant that provides very concise responses.
            Always respond in 1-2 sentences without any formatting, bullets, or quotes.
            Be direct and informative while maintaining a friendly tone."""
        )
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a text using OpenAI API."""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def _get_fallback_response(self, question: str) -> str:
        """Get a fallback response using GPT-4."""
        try:
            # Create a thread
            thread = client.beta.threads.create()
            
            # Add the user's question
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=question
            )
            
            # Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id
            )
            
            # Wait for completion
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == 'completed':
                    break
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    return "I apologize, but I'm not able to help with that right now."
                time.sleep(0.5)
            
            # Get the assistant's response
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            for message in messages.data:
                if message.role == "assistant":
                    return message.content[0].text.value
            
            return "I apologize, but I'm not able to help with that right now."
            
        except Exception as e:
            print(f"Fallback error: {str(e)}")
            return "I apologize, but I'm not able to help with that right now."
    
    def match_question(self, question: str) -> Dict[str, Any]:
        """Find the best matching intent for a question using ChromaDB."""
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(question)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=['metadatas', 'distances']
            )
            
            # Check if we have results
            if results['distances'] and results['metadatas']:
                # ChromaDB returns cosine distance, convert to similarity
                confidence = 1 - results['distances'][0][0]  # Convert distance to similarity
                
                # Check confidence threshold
                if confidence >= self.threshold:
                    metadata = results['metadatas'][0][0]
                    return {
                        "status": "success",
                        "intent": metadata["intent"],
                        "canonical_question": metadata["canonical_question"],
                        "confidence": confidence
                    }
            
            # If no match or below threshold, use fallback
            fallback_response = self._get_fallback_response(question)
            return {
                "status": "nlu_fallback",
                "confidence": confidence if results['distances'] else 0.0,
                "response": fallback_response
            }
            
        except Exception as e:
            print(f"Error in matching: {str(e)}")
            return {
                "status": "error",
                "confidence": 0.0,
                "response": "I apologize, but I'm having trouble processing your question."
            }

    def cleanup(self):
        """Cleanup resources."""
        try:
            client.beta.assistants.delete(self.assistant.id)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

def main():
    try:
        # Initialize matcher
        matcher = QuestionMatcher()
        
        print("\nQuestion Matching System")
        print("Enter your question (or 'quit' to exit)")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = matcher.match_question(question)
            
            if result["status"] == "success":
                print(f"Matched intent: {result['intent']}")
                print(f"Canonical question: {result['canonical_question']}")
                print(f"Confidence: {result['confidence']:.2f}")
            else:
                print(f"Status: {result['status']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Response: {result['response']}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Cleanup
        matcher.cleanup()

if __name__ == "__main__":
    main()
