from dotenv import load_dotenv
import os
from openai import OpenAI
import time

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()  # This will automatically use OPENAI_API_KEY from environment

def test_api():
    try:
        # Create a simple test thread
        thread = client.beta.threads.create()
        print(f"Created thread with ID: {thread.id}")

        # Create a simple test assistant
        assistant = client.beta.assistants.create(
            name="Test Assistant",
            model="gpt-4-1106-preview",
            instructions="You are a helpful assistant for testing."
        )
        print(f"Created assistant with ID: {assistant.id}")

        # Add a message to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Hi! Please give me a simple 'Hello, World!' response."
        )
        print("Added message to thread")

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        print(f"Started run with ID: {run.id}")

        # Wait for completion
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status == 'failed':
                print("Run failed!")
                return False
            time.sleep(1)
            print("Waiting for response...")

        # Get the messages
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        # Print all messages
        print("\nConversation:")
        for msg in messages.data:
            role = msg.role.capitalize()
            content = msg.content[0].text.value
            print(f"{role}: {content}")

        # Clean up
        client.beta.assistants.delete(assistant.id)
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
        exit(1)

    print("Testing OpenAI API connection...")
    success = test_api()
    
    if success:
        print("\nAPI test completed successfully!")
    else:
        print("\nAPI test failed!")
