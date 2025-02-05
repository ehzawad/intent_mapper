from dotenv import load_dotenv
import time
import chromadb
from openai import OpenAI
from typing import Dict, List, Any

# Load environment variables
load_dotenv()
client = OpenAI()

class QuestionMatcher:
    def __init__(self, collection_name: str = "intent_embeddings", threshold: float = 0.85):
        self.threshold = threshold
        self.batch_size = 10  # Maximum size for query batches
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get the collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"Connected to collection: {collection_name}")
            print(f"Collection size: {self.collection.count()} items")
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
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [data.embedding for data in response.data]
    
    def _get_fallback_response(self, question: str) -> str:
        """Get a fallback response using GPT-4."""
        try:
            thread = client.beta.threads.create()
            
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=question
            )
            
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id
            )
            
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == 'completed':
                    break
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    return "I apologize, but I'm not able to help with that right now."
                time.sleep(0.1)
            
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
    
    def match_questions_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Match multiple questions in batch."""
        try:
            # Get embeddings for all questions
            query_embeddings = self._get_embeddings_batch(questions)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=1,
                include=['metadatas', 'distances']
            )
            
            responses = []
            for i, (distances, metadatas) in enumerate(zip(results['distances'], results['metadatas'])):
                if distances and metadatas:
                    confidence = 1 - distances[0]  # Convert distance to similarity
                    
                    if confidence >= self.threshold:
                        metadata = metadatas[0]
                        responses.append({
                            "status": "success",
                            "intent": metadata["intent"],
                            "canonical_question": metadata["canonical_question"],
                            "confidence": confidence
                        })
                        continue
                
                # Fallback for this question
                fallback_response = self._get_fallback_response(questions[i])
                responses.append({
                    "status": "nlu_fallback",
                    "confidence": confidence if distances else 0.0,
                    "response": fallback_response
                })
            
            return responses
            
        except Exception as e:
            print(f"Error in batch matching: {str(e)}")
            return [{
                "status": "error",
                "confidence": 0.0,
                "response": "I apologize, but I'm having trouble processing your questions."
            } for _ in questions]
    
    def match_question(self, question: str) -> Dict[str, Any]:
        """Match a single question."""
        results = self.match_questions_batch([question])
        return results[0]

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
        print("For batch processing, enter multiple questions separated by '|'")
        
        while True:
            user_input = input("\nYour question(s): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Check if batch input
            if '|' in user_input:
                questions = [q.strip() for q in user_input.split('|')]
                results = matcher.match_questions_batch(questions)
                
                for i, result in enumerate(results):
                    print(f"\nQuestion {i+1}: {questions[i]}")
                    if result["status"] == "success":
                        print(f"Matched intent: {result['intent']}")
                        print(f"Canonical question: {result['canonical_question']}")
                        print(f"Confidence: {result['confidence']:.2f}")
                    else:
                        print(f"Status: {result['status']}")
                        print(f"Confidence: {result['confidence']:.2f}")
                        print(f"Response: {result['response']}")
            else:
                # Single question
                result = matcher.match_question(user_input)
                
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
