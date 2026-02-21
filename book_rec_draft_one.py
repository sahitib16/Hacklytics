import json
import time
import requests

movie_title = "Avengers: Endgame"

all_scenarios = [
    {
        "scenario_id": 1,
        "question": "What if Tony Stark had refused to help retrieve the Infinity Stones?",
        "context": "Tony had finally found peace with his family. What if he chose to stay retired, forcing the remaining Avengers to attempt the time heist without him?",
        "user_liked": None
    },
    {
        "scenario_id": 2,
        "question": "What if Natasha survived instead of Clint obtaining the Soul Stone?",
        "context": "In an alternate timeline, Clint sacrifices himself on Vormir instead of Natasha, leaving her to carry the grief and lead the restored Avengers.",
        "user_liked": None
    },
    {
        "scenario_id": 3,
        "question": "What if Steve Rogers never went back in time to return the Infinity Stones?",
        "context": "Instead of living out a quiet life in the past, Steve stays in the present — but what does that mean for his relationship with Peggy and his future with the team?",
        "user_liked": None
    },
    {
        "scenario_id": 4,
        "question": "What if Thanos had successfully retrieved all the Infinity Stones without any resistance?",
        "context": "The Avengers never reformed after the Snap. Thanos completed his mission unopposed — and now must govern the universe he reshaped.",
        "user_liked": None
    },
    {
        "scenario_id": 5,
        "question": "What if Thor had killed Thanos successfully at the beginning of Endgame?",
        "context": "Thor's axe lands true and Thanos dies — but the Stones are already destroyed. The Snap is permanent, and the Avengers must learn to live in a world where victory feels hollow.",
        "user_liked": None
    },
    {
        "scenario_id": 6,
        "question": "What if the Hulk had failed to bring everyone back with his Snap?",
        "context": "Bruce Banner attempts the reverse Snap but something goes wrong — only some of the vanished return, leaving the world in chaos as families are torn between two realities.",
        "user_liked": None
    },
    {
        "scenario_id": 7,
        "question": "What if Nebula had not been discovered as a traitor and successfully handed the Infinity Gauntlet to Thanos?",
        "context": "Past-Nebula's infiltration goes undetected. Thanos arrives in the present with a fully functional Gauntlet, and the Avengers face him with no plan and no backup.",
        "user_liked": None
    },
    {
        "scenario_id": 8,
        "question": "What if Captain America had chosen to stay in the past permanently with Peggy?",
        "context": "Steve makes his choice and lives out his life in the 1940s — but his absence in the present leaves a void that no one can fill, and a young Bucky with no one to save him.",
        "user_liked": None
    },
    {
        "scenario_id": 9,
        "question": "What if Peter Parker had not been brought back by the Snap and Tony had to live knowing Spider-Man was gone forever?",
        "context": "Peter remains among the vanished even after the Hulk's Snap due to an unforeseen anomaly. Tony must continue as Iron Man carrying the weight of a loss that feels deeply personal.",
        "user_liked": None
    },
    {
        "scenario_id": 10,
        "question": "What if Wanda had killed Thanos before his army arrived?",
        "context": "Scarlet Witch's power is enough to destroy Thanos alone — but his army still arrives through the portals, leaderless and rampaging, with no one left to call them off.",
        "user_liked": None
    }
]

hardcoded_queries = {
    1: ["The Road Cormac McCarthy", "A Man Called Ove Fredrik Backman"],
    2: ["The Lovely Bones Alice Sebold", "Extremely Loud and Incredibly Close Jonathan Safran Foer"],
    3: ["The Time Traveler's Wife Audrey Niffenegger", "Kindred Octavia Butler"],
    4: ["The Giver Lois Lowry", "Brave New World Aldous Huxley"],
    5: ["All Quiet on the Western Front Erich Maria Remarque", "The Things They Carried Tim O'Brien"],
    6: ["Station Eleven Emily St John Mandel", "The Stand Stephen King"],
    7: ["Tinker Tailor Soldier Spy John le Carre", "The Spy Who Came in from the Cold John le Carre"],
    8: ["The Notebook Nicholas Sparks", "Outlander Diana Gabaldon"],
    9: ["A Little Life Hanya Yanagihara", "The Art of Racing in the Rain Garth Stein"],
    10: ["Carrie Stephen King", "The Power Naomi Alderman"]
}

def get_user_input():
    print("Alternative Universe Scenario Generator\n")
    print(f"Movie: {movie_title}\n")

    while True:
        try:
            num_prompts = int(input("How many scenarios would you like? Enter a number from 1-10: ").strip())
            if 1 <= num_prompts <= 10:
                break
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    return num_prompts

def collect_user_responses(selected_scenarios):
    results = []

    for scenario in selected_scenarios:
        print(f"\nScenario {scenario['scenario_id']}")
        print(f"QUESTION: {scenario['question']}")
        print(f"CONTEXT:  {scenario['context']}")

        while True:
            answer = input("\nWould you enjoy this alternate universe? Answer yes or no: ").strip().lower()
            if answer in ("yes", "no", "y", "n"):
                liked = answer in ("yes", "y")
                break
            print("Please enter 'yes' or 'no'.")

        scenario["user_liked"] = liked
        results.append(scenario)
        print(f"Recorded: {'Liked' if liked else 'Disliked'}")

    return results

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
            "info_link": f"https://openlibrary.org{book.get('key', '')}",
            "search_query": query
        }

    except requests.RequestException as e:
        print(f"  Open Library error: {e}")
        return None

def get_books_for_liked_scenarios(liked_scenarios):
    """For each liked scenario, searches Open Library using hardcoded queries."""
    all_results = []
    seen_titles = set()

    for scenario in liked_scenarios:
        scenario_id = scenario["scenario_id"]
        print(f"\nProcessing Scenario {scenario_id}: {scenario['question']}")

        queries = hardcoded_queries.get(scenario_id, [])
        if not queries:
            print(f"  No queries found for scenario {scenario_id}")
            continue

        scenario_books = []

        for query in queries[:2]:
            time.sleep(0.5)
            print(f"  Searching Open Library: '{query}'...")
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
            "scenario_id": scenario_id,
            "scenario_question": scenario["question"],
            "books": scenario_books
        })

    return all_results

print("Starting Multiverse Generator...")

num_prompts = get_user_input()

selected_scenarios = all_scenarios[:num_prompts]

user_responses = collect_user_responses(selected_scenarios)
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
