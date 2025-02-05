from dotenv import load_dotenv
import time
import asyncio
import chromadb
from openai import AsyncOpenAI
from typing import Dict, Any
import cachetools

# Load environment variables
load_dotenv()
client = AsyncOpenAI()

class FastQuestionMatcher:
    def __init__(self, collection_name: str = "intent_embeddings", threshold: float = 0.85):
        self.threshold = threshold
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_collection(collection_name)
        print(f"Connected to collection: {collection_name}")
        
        # Cache for embeddings
        self.embedding_cache = cachetools.TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        
        # Initialize assistant
        self.assistant = None  # Will be initialized on first use
        
    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding with caching."""
        cache_key = text.strip().lower()
        
        # Check cache first
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Get new embedding
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response.data[0].embedding
        
        # Cache it
        self.embedding_cache[cache_key] = embedding
        return embedding

    async def _ensure_assistant(self):
        """Lazily initialize the assistant."""
        if self.assistant is None:
            self.assistant = await client.beta.assistants.create(
                name="Fallback Handler",
                model="gpt-4-1106-preview",
                instructions="""You are a helpful assistant that provides very concise responses.
                Always respond in 1-2 sentences without any formatting."""
            )

    async def _get_fallback_response(self, question: str) -> str:
        """Get fallback response using Assistant API."""
        try:
            await self._ensure_assistant()
            
            # Create thread and add message
            thread = await client.beta.threads.create()
            await client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=question
            )
            
            # Run the assistant
            run = await client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id
            )
            
            # Check status with timeout
            start_time = time.time()
            while time.time() - start_time < 5:  # 5 second timeout
                run_status = await client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == 'completed':
                    messages = await client.beta.threads.messages.list(
                        thread_id=thread.id
                    )
                    for msg in messages.data:
                        if msg.role == "assistant":
                            return msg.content[0].text.value
                    break
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    break
                await asyncio.sleep(0.1)
            
            return "I apologize, but I'm not able to help with that right now."
            
        except Exception as e:
            print(f"Fallback error: {str(e)}")
            return "I apologize, but I'm not able to help with that right now."

    async def match_question(self, question: str) -> Dict[str, Any]:
        """Find the best matching intent for a question."""
        try:
            # Get embedding
            query_embedding = await self._get_embedding(question)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=['metadatas', 'distances']
            )
            
            # Process results
            if results['distances'] and results['metadatas']:
                confidence = 1 - results['distances'][0][0]  # Convert distance to similarity
                
                if confidence >= self.threshold:
                    metadata = results['metadatas'][0][0]
                    return {
                        "status": "success",
                        "intent": metadata["intent"],
                        "canonical_question": metadata["canonical_question"],
                        "confidence": confidence
                    }
            
            # If no match or below threshold, get fallback
            fallback_response = await self._get_fallback_response(question)
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

    async def cleanup(self):
        """Cleanup resources."""
        if self.assistant:
            try:
                await client.beta.assistants.delete(self.assistant.id)
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

async def main():
    matcher = FastQuestionMatcher()
    
    try:
        print("\nFast Question Matching System")
        print("Enter your question (or 'quit' to exit)")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            start_time = time.time()
            result = await matcher.match_question(question)
            end_time = time.time()
            
            if result["status"] == "success":
                print(f"Matched intent: {result['intent']}")
                print(f"Canonical question: {result['canonical_question']}")
                print(f"Confidence: {result['confidence']:.2f}")
            else:
                print(f"Status: {result['status']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Response: {result['response']}")
            
            print(f"Response time: {(end_time - start_time):.3f} seconds")
    
    finally:
        await matcher.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
