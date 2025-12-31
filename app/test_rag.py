# test_rag.py
from rag import generate_answer

def main():
    # Example query
    query = "Which cities have the highest PM2.5 pollution?"

    try:
        answer = generate_answer(query)
        print("Query:", query)
        print("Answer:\n", answer)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
