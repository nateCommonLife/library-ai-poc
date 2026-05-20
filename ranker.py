import os
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

def get_abstract(abstract_inverted_index):
    """Reconstructs abstract from OpenAlex's inverted index format."""
    if not abstract_inverted_index:
        return "No abstract available."
    
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)

def fetch_works(research_context):
    """Fetches works from OpenAlex based on research context."""
    ANCIENT_ROME_ID = "C528414297"
    CLASSICS_ID = "C74916050"
    
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"concepts.id:{ANCIENT_ROME_ID}|{CLASSICS_ID}",
        "search": research_context,
        "per_page": 15,
        "select": "id,title,publication_year,cited_by_count,referenced_works,authorships,abstract_inverted_index",
    }
    
    response = requests.get(url, params=params)
    print("Raw API response:", response.json())
    return response.json()["results"]


def rank_and_characterize(research_context, works):
    """Uses Claude to re-rank and characterize works by relevance."""
    
    # Build a summary of works to send to Claude
    works_summary = []
    for i, work in enumerate(works):
        author = "Unknown"
        if work.get("authorships"):
            author = work["authorships"][0]["author"]["display_name"]
        
        abstract = get_abstract(work.get("abstract_inverted_index"))
        
        works_summary.append(
            f"[{i}] Title: {work.get('title', 'Unknown')}\n"
            f"    Author: {author} | Year: {work.get('publication_year', 'Unknown')} | "
            f"Cited by: {work.get('cited_by_count', 0)}\n"
            f"    Abstract: {abstract[:300]}..."
        )
    
    works_text = "\n\n".join(works_summary)
    
    prompt = f"""You are an expert academic research librarian helping a scholar find relevant literature.

The researcher's context is:
"{research_context}"

Here are {len(works)} works retrieved from an academic database. For each work:
1. Score its relevance to the research context (0-10)
2. Briefly characterize what role it likely plays (e.g. "Primary source on Roman family law", "Methodological framework", "Tangentially related", "Not relevant")
3. Note if it appears to represent a distinct scholarly position or school of thought

Works:
{works_text}

Respond in this exact format for each work:
[INDEX] Score: X/10 | Role: [characterization] | Notes: [1-2 sentences max]

After scoring all works, list the top 5 by relevance in order, with a one-sentence explanation of why each is valuable to this researcher.

Be honest — if works are not relevant, say so. The researcher is counting on your accuracy."""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text

def search_and_rank(research_context):
    """Main function - fetches and ranks works for a research context."""
    print(f"\nResearch context: {research_context}")
    print("=" * 60)
    
    print("\nFetching works from OpenAlex...")
    works = fetch_works(research_context)
    print(f"Retrieved {len(works)} works")
    
    print("\nAsking Claude to analyze and rank results...")
    analysis = rank_and_characterize(research_context, works)
    
    print("\n--- CLAUDE'S ANALYSIS ---\n")
    print(analysis)
    
    return works, analysis

if __name__ == "__main__":
    search_and_rank(
        "I am writing a dissertation on adoption practices in the first century Roman world. "
        "I need scholarly works on Roman family structure, legal adoption, patria potestas, "
        "and social practices around inheritance and family formation in antiquity."
    )