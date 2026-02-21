import os
import json
from typing import Any

from dotenv import load_dotenv
import google.generativeai as genai

from scrape_data_reddit import ActianCortexStore, LocalHasherEmbedder, semantic_search, env_int


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def get_user_input():
    print("Alternative Universe Scenario Generator (RAG)\n")

    movie_title = input("Enter the movie/TV show title: ").strip()
    while not movie_title:
        print("Title cannot be empty. Please try again.")
        movie_title = input("Enter the movie/TV show title: ").strip()

    while True:
        try:
            num_prompts = int(input("How many scenarios would you like? Enter a number from 1-10: ").strip())
            if 1 <= num_prompts <= 10:
                break
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    return movie_title, num_prompts


def retrieve_context(movie_title: str, top_k: int = 8) -> list[dict[str, Any]]:
    local_embed_dim = env_int("LOCAL_EMBED_DIM", 1024)
    cortex_address = os.getenv("CORTEX_ADDRESS", "localhost:50051")
    collection_name = os.getenv("ACTIAN_COLLECTION_NAME", "endgame_opinions")

    store = ActianCortexStore(
        address=cortex_address,
        collection_name=collection_name,
        dimension=local_embed_dim,
        recreate=False,
    )
    embedder = LocalHasherEmbedder(dim=local_embed_dim)
    query = f"{movie_title} reviews audience opinions controversies"
    return semantic_search(store=store, embedder=embedder, query=query, top_k=top_k, source_filter=None)


def build_context(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "No context retrieved from vector database."
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] source={hit.get('source')} title={hit.get('title')} url={hit.get('url')}\n{hit.get('text', '')}"
        )
    return "\n\n".join(blocks)


def get_au_scenario(movie_title, num_prompts):
    print(f"\nGenerating {num_prompts} scenario(s) for: {movie_title}")

    hits = retrieve_context(movie_title=movie_title, top_k=8)
    rag_context = build_context(hits)

    prompt = f"""
    You are generating controversial "What If" scenarios grounded in retrieved audience-opinion context.

    Movie/TV show: "{movie_title}"

    Retrieved context from vector DB:
    {rag_context}

    Generate {num_prompts} highly controversial 'What If' scenarios for "{movie_title}".
    Each question should incite a Yes or No answer from the user.
    Use the retrieved context to make scenarios more specific and debatable.
    Return ONLY a JSON array with exactly {num_prompts} objects:
    [
        {{
            "question": "The question",
            "context": "The explanation"
        }}
    ]
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )

        data = json.loads(response.text)

        for i, scenario in enumerate(data, 1):
            print(f"\nScenario {i}:")
            print(f" QUESTION:{scenario['question']}")
            print(f" CONTEXT:{scenario['context']}")

        return data

    except Exception as e:
        print(f"Error generating scenarios for {movie_title}: {e}")
        return None


print("Starting Multiverse Generator (RAG)...")
movie_title, num_prompts = get_user_input()
results = get_au_scenario(movie_title, num_prompts)
if results:
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to results.json")
else:
    print("\nNo results to save.")
