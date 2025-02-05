from dotenv import load_dotenv
import yaml
from openai import OpenAI
import time

# Load environment variables
load_dotenv()
client = OpenAI()

def load_intents(file_path: str) -> dict:
    """Load intents from YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text using OpenAI API."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def calculate_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Simple dot product for similarity
    return sum(a * b for a, b in zip(embedding1, embedding2))

def find_matching_intent(query: str, intents: dict) -> tuple[str, str, float] | None:
    """Find the best matching intent for a given query."""
    query_embedding = get_embedding(query)
    best_match = None
    best_score = 0
    best_canonical = None

    for intent_name, intent_data in intents['intents'].items():
        # Check canonical question and variations
        all_questions = [intent_data['canonical_question']] + intent_data['variations']
        
        for question in all_questions:
            question_embedding = get_embedding(question)
            similarity = calculate_similarity(query_embedding, question_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = intent_name
                best_canonical = intent_data['canonical_question']

    if best_score > 0.8:  # Threshold for matching
        return (best_match, best_canonical, best_score)
    return None

def main():
    # Load intents
    intents = load_intents('intents.yaml')
    
    # Test questions
    test_questions = [
        "How much do I have in my account?",
        "Is my card working properly?",
        "What's my current balance?",
        "Can you check my card status?",
    ]

    print("Testing question matching:")
    for question in test_questions:
        print(f"\nUser asks: {question}")
        match = find_matching_intent(question, intents)
        if match:
            intent_name, canonical_question, confidence = match
            print(f"Matched intent: {intent_name}")
            print(f"Canonical question: {canonical_question}")
            print(f"Confidence: {confidence:.2f}")
        else:
            print("No match found")

if __name__ == "__main__":
    main()
