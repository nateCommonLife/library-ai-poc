import requests
import json

def search_openalex(research_context):
    """
    Takes a plain-language research context and queries OpenAlex
    using the Ancient Rome concept filter, returning relevant works.
    """
    
    ANCIENT_ROME_ID = "C528414297"
    CLASSICS_ID = "C74916050"
    
    print(f"\nSearching for: {research_context}")
    print("-" * 60)
    
    # Query OpenAlex with concept filter + keyword search
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"concepts.id:{ANCIENT_ROME_ID}",
        "search": research_context,
        "per_page": 10,
        "select": "id,title,publication_year,cited_by_count,referenced_works,authorships,abstract_inverted_index",
        "sort": "cited_by_count:desc"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    total = data["meta"]["count"]
    print(f"Total matching works: {total}\n")
    
    for i, work in enumerate(data["results"], 1):
        title = work.get("title", "No title")
        year = work.get("publication_year", "Unknown")
        citations = work.get("cited_by_count", 0)
        refs = len(work.get("referenced_works", []))
        
        # Get first author if available
        authorships = work.get("authorships", [])
        author = authorships[0]["author"]["display_name"] if authorships else "Unknown"
        
        print(f"{i}. {title}")
        print(f"   Author: {author} | Year: {year} | Cited by: {citations} | References: {refs}")
        print(f"   ID: {work['id']}")
        print()
    
    return data["results"]

if __name__ == "__main__":
    results = search_openalex("adoption practices family structure Roman antiquity")