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

def identify_key_scholars(research_context):
    """
    Asks Claude to identify the foundational scholars for a research topic.
    Returns a list of scholar names and key search terms.
    """
    print("\nStep 1: Identifying key scholars and search terms...")
    
    prompt = f"""You are an expert academic research librarian with deep knowledge of scholarly literature.

A researcher has provided this context:
"{research_context}"

Your task:
1. Identify 5-8 foundational or prominent scholars who have published significantly on this topic
2. Identify 5-8 precise academic search terms or phrases that would find the most relevant literature
3. Identify 2-3 opposing scholarly positions or schools of thought on this topic if they exist

Respond in this exact JSON format with no additional text:
{{
    "scholars": ["Scholar Name 1", "Scholar Name 2", "Scholar Name 3"],
    "search_terms": ["term 1", "term 2", "term 3"],
    "opposing_positions": [
        {{"position": "description of position", "proponents": ["Scholar A", "Scholar B"]}},
        {{"position": "description of opposing position", "proponents": ["Scholar C", "Scholar D"]}}
    ]
}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    import json
    response_text = message.content[0].text.strip()
    # Strip markdown code fences if present
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    return json.loads(response_text.strip())

def search_by_author(author_name):
    """Searches OpenAlex for works by a specific author."""
    url = "https://api.openalex.org/works"
    params = {
        "search": author_name,
        "per_page": 5,
        "select": "id,title,publication_year,cited_by_count,referenced_works,authorships,abstract_inverted_index",
        "sort": "cited_by_count:desc"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("results", [])

def search_by_term(term):
    """Searches OpenAlex by a specific academic term."""
    ANCIENT_ROME_ID = "C528414297"
    CLASSICS_ID = "C74916050"
    
    url = "https://api.openalex.org/works"
    params = {
        "search": term,
        "per_page": 5,
        "select": "id,title,publication_year,cited_by_count,referenced_works,authorships,abstract_inverted_index",
        "sort": "cited_by_count:desc"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("results", [])

def deduplicate_works(all_works):
    """Removes duplicate works by ID."""
    seen_ids = set()
    unique_works = []
    for work in all_works:
        work_id = work.get("id")
        if work_id and work_id not in seen_ids:
            seen_ids.add(work_id)
            unique_works.append(work)
    return unique_works

def rank_and_characterize(research_context, works, opposing_positions):
    """Uses Claude to re-rank and characterize works by relevance."""
    
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
    opposing_text = "\n".join([
        f"- {p['position']} (proponents: {', '.join(p['proponents'])})"
        for p in opposing_positions
    ])

    prompt = f"""You are an expert academic research librarian.

The researcher's context:
"{research_context}"

Known opposing scholarly positions on this topic:
{opposing_text}

Here are {len(works)} works retrieved from academic databases. For each work:
1. Score relevance to the research context (0-10)
2. Characterize its role (e.g. "Foundational text on Roman adoption law", "Methodological framework", "Not relevant")
3. Note which scholarly position it represents if applicable

Works:
{works_text}

Respond in this exact format for each work:
[INDEX] Score: X/10 | Role: [characterization] | Position: [scholarly position or "N/A"] | Notes: [1-2 sentences]

Then provide:
LAYER 1 - TOP RELEVANT WORKS (score 6+), ranked by relevance
LAYER 2 - INTELLECTUAL LINEAGE: Which of these works cite foundational texts? What earlier works shaped this field?
LAYER 3 - SCHOLARLY DEBATE: Which works represent opposing positions? What is the core disagreement?"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def scholar_seeded_search(research_context):
    """
    Main function - uses Claude's knowledge to seed better searches,
    then re-ranks and presents results in three layers.
    """
    print(f"\nResearch context: {research_context}")
    print("=" * 60)

    # Step 1 - Claude identifies scholars, terms, opposing positions
    intelligence = identify_key_scholars(research_context)
    
    print(f"\nKey scholars identified: {', '.join(intelligence['scholars'])}")
    print(f"Search terms identified: {', '.join(intelligence['search_terms'])}")
    print(f"Opposing positions found: {len(intelligence['opposing_positions'])}")

    # Step 2 - Search OpenAlex by each scholar and term
    all_works = []
    
    print("\nStep 2: Searching by scholar name...")
    for scholar in intelligence["scholars"]:
        works = search_by_author(scholar)
        if works:
            print(f"  {scholar}: {len(works)} works found")
            all_works.extend(works)
        else:
            print(f"  {scholar}: no works found in OpenAlex")

    print("\nSearching by academic terms...")
    for term in intelligence["search_terms"]:
        works = search_by_term(term)
        if works:
            print(f"  '{term}': {len(works)} works found")
            all_works.extend(works)

    # Step 3 - Deduplicate
    unique_works = deduplicate_works(all_works)
    print(f"\nTotal unique works retrieved: {len(unique_works)}")

    if not unique_works:
        print("No works found. Consider adding Semantic Scholar as a data source.")
        return

    # Step 4 - Claude ranks and presents in three layers
    print("\nStep 3: Claude analyzing and organizing into three layers...")
    analysis = rank_and_characterize(
        research_context,
        unique_works[:20],  # Cap at 20 to keep prompt manageable
        intelligence["opposing_positions"]
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(analysis)

    return intelligence, unique_works, analysis

if __name__ == "__main__":
    scholar_seeded_search(
        "I am writing a dissertation on adoption practices in the first century Roman world. "
        "I need scholarly works on Roman family structure, legal adoption, patria potestas, "
        "and social practices around inheritance and family formation in antiquity."
    )