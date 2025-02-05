from dotenv import load_dotenv
import os
from openai import OpenAI
import time

# Load environment variables
load_dotenv()
client = OpenAI()

def generate_question_variations(original_question: str) -> list[str]:
    """Generate variations of a given question using GPT-4."""
    
    # Create a new thread
    thread = client.beta.threads.create()
    
    # Create our prompt
    prompt = f"""Please generate 5 different ways to ask this question:
    Original: "{original_question}"
    
    Generate natural variations that ask for the same information. Each should end with a question mark.
    Provide them one per line, without numbering."""
    
    # Add the message to thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    
    # Create an assistant for this task
    assistant = client.beta.assistants.create(
        name="Question Variation Generator",
        model="gpt-4-1106-preview",
        instructions="You generate different ways to ask the same question. Be natural and conversational."
    )
    
    try:
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for completion
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status == 'failed':
                print("Failed to generate variations")
                return []
            time.sleep(1)
        
        # Get the response
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        # Extract variations from the assistant's response
        variations = []
        for msg in messages.data:
            if msg.role == "assistant":
                # Split the response into lines and clean them up
                lines = msg.content[0].text.value.strip().split('\n')
                variations.extend([line.strip() for line in lines if '?' in line])
        
        return variations
        
    finally:
        # Clean up
        client.beta.assistants.delete(assistant.id)

def match_question(user_question: str, canonical_questions: dict) -> tuple[str, float] | None:
    """Find the best matching canonical question using embeddings."""
    
    # Get embedding for user question
    user_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=user_question
    ).data[0].embedding
    
    best_match = None
    best_similarity = 0
    
    # Compare with each canonical question and its variations
    for question_id, variations in canonical_questions.items():
        for variation in variations:
            # Get embedding for this variation
            variation_embedding = client.embeddings.create(
                model="text-embedding-ada-002",
                input=variation
            ).data[0].embedding
            
            # Calculate similarity using dot product
            similarity = sum(a * b for a, b in zip(user_embedding, variation_embedding))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = question_id
    
    if best_similarity > 0.8:  # Threshold for matching
        return (best_match, best_similarity)
    return None

def main():
    # Example canonical questions with their IDs
    canonical_questions = {
        "balance": ["What is my account balance?"],
        "transfer": ["How do I transfer money between accounts?"],
    }
    
    # Generate variations for each canonical question
    for question_id, questions in canonical_questions.items():
        print(f"\nGenerating variations for: {questions[0]}")
        variations = generate_question_variations(questions[0])
        canonical_questions[question_id].extend(variations)
        print("Variations generated:")
        for var in variations:
            print(f"- {var}")
    
    # Test some questions
    test_questions = [
        "How much money do I have?",
        "Can you show me my balance?",
        "I want to move some money around",
        "What's the process for sending money to another account?"
    ]
    
    print("\nTesting question matching:")
    for question in test_questions:
        print(f"\nUser asks: {question}")
        match = match_question(question, canonical_questions)
        if match:
            question_id, confidence = match
            print(f"Matched to: {question_id}")
            print(f"Confidence: {confidence:.2f}")
        else:
            print("No match found")

if __name__ == "__main__":
    main()
