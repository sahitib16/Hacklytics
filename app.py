import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

def get_au_scenario(movie_title, num_prompts):
    print(f"\nGenerating {num_prompts} scenario(s) for: {movie_title}")
    
    prompt = f"""
    Generate {num_prompts} unique and highly controversial 'What If' scenarios for "{movie_title}".
    Each question should incite a Yes or No answer from the user.
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
            generation_config={"response_mime_type": "application/json"}
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

print("Starting Multiverse Generator...")
movie_title, num_prompts = get_user_input()
results = get_au_scenario(movie_title, num_prompts)
if results:
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to results.json")
else:
    print("\nNo results to save.")
