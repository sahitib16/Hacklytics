import os
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_user_input():
    print("Alternative Universe Scenario Generator\n")

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

def get_au_scenario(movie_title, num_prompts):
    print(f"\nGenerating {num_prompts} scenario(s) for: {movie_title}")

    prompt = f"""
    Generate {num_prompts} controversial but popular 'What If' scenarios for "{movie_title}".
    Each question should incite a Yes/No answer from the user.
    Return ONLY a JSON array with exactly {num_prompts} objects:
    [
        {{
            "question": "The question",
            "context": "The explanation"
        }}
    ]
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)

    except Exception as e:
        print(f"Error generating scenarios for {movie_title}: {e}")
        return None

def collect_user_responses(scenarios):
    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}")
        print(f"QUESTION: {scenario['question']}")
        print(f"CONTEXT: {scenario['context']}")

        while True:
            answer = input("\nWould you enjoy this alternate universe? Answer yes or no: ").strip().lower()
            if answer in ("yes", "no", "y", "n"):
                liked = answer in ("yes", "y")
                break
            print("Please enter 'yes' or 'no'.")

        results.append({
            "scenario_id": i,
            "question": scenario["question"],
            "context": scenario["context"],
            "user_liked": liked
        })

        print(f"Recorded: {'Liked' if liked else 'Disliked'}")

    return results

def get_queries_for_scenario(scenario):
    """
    Uses Gemini with Google Search grounding to find real book titles
    matching the scenario's themes, then returns them as Open Library queries.
    """
    prompt = f"""
    A user liked this alternate universe scenario:

    Question: {scenario['question']}
    Theme: {scenario['context']}

    Step 1: Identify the core emotional and narrative themes in this scenario.
    For example: "a hero sacrificing peace for duty", "survivor's guilt after losing a friend".

    Step 2: Use Google Search to find exactly 2 real, published books that strongly
    match these themes. Look for well-known novels or literary fiction — not comics,
    screenplays, or fanfiction. The books must actually exist and be findable on
    Open Library (openlibrary.org).

    Step 3: Return the 2 book titles and authors as search query strings in the format
    "Book Title Author Name" so they can be searched directly on Open Library.

    Return ONLY a JSON array with exactly 2 search query strings and nothing else:
    ["Book Title One Author One", "Book Title Two Author Two"]
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        raw_text = response.text.strip()

        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        queries = json.loads(raw_text.strip())
        print(f"  Gemini grounded queries for scenario {scenario['scenario_id']}: {queries}")
        return queries

    except Exception as e:
        print(f"  Error generating queries for scenario {scenario['scenario_id']}: {e}")
        return []

def search_open_library(query):
    """Searches Open Library by title+author and returns the best matching book."""
    url = "https://openlibrary.org/search.json"
    params = {
        "q": query,
        "limit": 1,
        "lang": "eng"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("docs"):
            return None

        book = data["docs"][0]

        return {
            "title": book.get("title", "N/A"),
            "authors": book.get("author_name", ["N/A"]),
            "first_published": book.get("first_publish_year", "N/A"),
            "subjects": book.get("subject", [])[:5],
            "info_link": f"https://openlibrary.org{book.get('key', '')}",
            "search_query": query
        }

    except requests.RequestException as e:
        print(f"  Open Library error: {e}")
        return None

def get_books_for_liked_scenarios(liked_scenarios):
    """For each liked scenario, gets 2 grounded Gemini queries and searches Open Library."""
    all_results = []
    seen_titles = set()

    for scenario in liked_scenarios:
        print(f"\nProcessing Scenario {scenario['scenario_id']}: {scenario['question']}")
        queries = get_queries_for_scenario(scenario)
        scenario_books = []

        for query in queries[:2]:
            time.sleep(0.5)
            print(f"Searching Open Library: '{query}'...")
            result = search_open_library(query)

            if result and result["title"] not in seen_titles:
                seen_titles.add(result["title"])
                scenario_books.append(result)
                print(f"Found: '{result['title']}' by {', '.join(result['authors'])}")
            elif result and result["title"] in seen_titles:
                print(f"Duplicate skipped: '{result['title']}'")
            else:
                print(f"No result found for: '{query}'")

        all_results.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_question": scenario["question"],
            "books": scenario_books
        })

    return all_results

print("Starting Multiverse Generator...")

movie_title, num_prompts = get_user_input()
scenarios = get_au_scenario(movie_title, num_prompts)

if not scenarios:
    print("Failed to generate scenarios. Exiting.")
    exit()

user_responses = collect_user_responses(scenarios)
liked_scenarios = [s for s in user_responses if s["user_liked"]]

print(f"\nYou liked {len(liked_scenarios)} out of {num_prompts} scenario(s).")

if not liked_scenarios:
    print("No liked scenarios. No book recommendations to generate.")
    book_recommendations = []
else:
    book_recommendations = get_books_for_liked_scenarios(liked_scenarios)

output = {
    "movie_title": movie_title,
    "scenarios": user_responses,
    "liked_count": len(liked_scenarios),
    "disliked_count": num_prompts - len(liked_scenarios),
    "book_recommendations": book_recommendations
}

with open('results.json', 'w') as f:
    json.dump(output, f, indent=4)

print("\nResults saved to results.json")
