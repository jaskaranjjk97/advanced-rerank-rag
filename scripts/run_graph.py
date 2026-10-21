from app.graph import compiled_graph


def main():
    query = "What is the trend in direct tax collections for FY26?"
    print(f"Query: {query}\n")
    print("Running pipeline (Retrieve -> Rerank -> Generate)...\n")

    result = compiled_graph.invoke({"query": query})
    print("=== FINAL ANSWER ===")
    print(result["final_response"])

    print("\n=== TRACE ===")
    for step in result["trace"]:
        print(f"[{step['node'].upper()}] {step['details']}")


if __name__ == "__main__":
    main()
