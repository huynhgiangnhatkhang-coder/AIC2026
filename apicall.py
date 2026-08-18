import sys

try:
    from pymilvus import MilvusClient
    from openai import OpenAI
except ImportError:
    print("Please install the required libraries: pip install pymilvus openai")
    sys.exit(1)

def check_milvus():
    print("--- Checking Milvus Connection ---")
    try:
        # Connect to your Milvus database
        client = MilvusClient(uri="http://aicpc.sytes.net:19530")
        
        # Check connection by fetching existing collections
        collections = client.list_collections()
        
        print("Success: Connected to Milvus!")
        print(f"Collections found: {collections}")
        
    except Exception as e:
        print(f"Error: Failed to connect to Milvus.\nDetails: {e}")

def check_lmstudio():
    print("\n--- Checking LM Studio Connection ---")
    try:
        # Initialize the OpenAI client pointing to your LM Studio server
        client = OpenAI(
            base_url="http://aicpc.sytes.net:1234/v1",
            api_key="lm-studio"
        )
        
        print("Sending query 'Who are you?' to your local model...")
        
        # Generate a response
        completion = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "user", "content": "Who are you?"}
            ],
            temperature=0.7
        )
        
        print("Success: Connected to LM Studio!")
        print("\nModel Response:")
        print("-" * 40)
        print(completion.choices[0].message.content)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error: Failed to connect to LM Studio.\nDetails: {e}")

if __name__ == "__main__":
    check_milvus()
    check_lmstudio()