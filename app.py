from flask import Flask, render_template, request, Response, stream_with_context
import json
import requests
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
client = Anthropic()

def get_abstract(work):
    if work.get("abstract_text"):
        return work["abstract_text"]
    abstract_inverted_index = work.get("abstract_inverted_index")
    if not abstract_inverted_index:
        return "No abstract available."
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)

def identify_key_scholars(research_context):
    prompt = f"""You are an expert academic research librarian with deep knowledge of scholarly literature.

A researcher has provided this context:
"{research_context}"

Your task:
1. Identify 5-8 foundational or prominent scholars who have published significantly on this topic
2. Identify 5-8 precise academic search terms or phrases that would find the most relevant literature
3. Identify 2-3 opposing scholarly positions or schools of thought on this topic if they exist

Respond in this exact JSON format with no additional text and no markdown fences:
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
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    return json.loads(response_text.strip())

def search_by_author(author_name):
    url = "https://api.openalex.org/works"
    params = {
        "search": author_name,
        "per_page": 5,
        "select": "id,title,publication_year,cited_by_count,referenced_works,authorships,abstract_inverted_index",
        "sort": "cited_by_count:desc"
    }
    response = requests.get(url, params=params)
    results = response.json().get("results", [])
    for r in results:
        r["source"] = "OpenAlex"
        r["abstract_text"] = None
    return results

def search_by_term(term):
    url = "https://api.openalex.org/works"
    params = {
        "search": term,
        "per_page": 5,
        "select": "id,title,publication_year,cited_by_count,referenced_works,authorships,abstract_inverted_index",
        "sort": "cited_by_count:desc"
    }
    response = requests.get(url, params=params)
    results = response.json().get("results", [])
    for r in results:
        r["source"] = "OpenAlex"
        r["abstract_text"] = None
    return results

def normalize_semantic_scholar(papers):
    normalized = []
    for p in papers:
        authors = p.get("authors", [])
        normalized.append({
            "id": f"ss:{p.get('paperId', '')}",
            "title": p.get("title"),
            "publication_year": p.get("year"),
            "cited_by_count": p.get("citationCount", 0),
            "referenced_works": [
                f"ss:{r['paperId']}"
                for r in p.get("references", [])
                if r.get("paperId")
            ],
            "authorships": [
                {"author": {"display_name": a.get("name", "Unknown")}}
                for a in authors
            ],
            "abstract_inverted_index": None,
            "abstract_text": p.get("abstract", "No abstract available."),
            "source": "Semantic Scholar"
        })
    return normalized

def search_semantic_scholar_by_author(author_name):
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": author_name,
        "limit": 5,
        "fields": "title,year,citationCount,abstract,authors,references"
    }
    headers = {"x-api-key": api_key}
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        return normalize_semantic_scholar(data.get("data", []))
    except Exception:
        return []

def search_semantic_scholar_by_term(term):
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": term,
        "limit": 5,
        "fields": "title,year,citationCount,abstract,authors,references"
    }
    headers = {"x-api-key": api_key}
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        return normalize_semantic_scholar(data.get("data", []))
    except Exception:
        return []

def deduplicate_works(all_works):
    seen_ids = set()
    unique_works = []
    for work in all_works:
        work_id = work.get("id")
        if work_id and work_id not in seen_ids:
            seen_ids.add(work_id)
            unique_works.append(work)
    return unique_works

def build_analysis_prompt(research_context, works, opposing_positions):
    works_summary = []
    for i, work in enumerate(works):
        author = "Unknown"
        if work.get("authorships"):
            author = work["authorships"][0]["author"]["display_name"]
        abstract = get_abstract(work)
        works_summary.append(
            f"[{i}] Title: {work.get('title', 'Unknown')}\n"
            f"    Author: {author} | Year: {work.get('publication_year', 'Unknown')} | "
            f"Cited by: {work.get('cited_by_count', 0)} | Source: {work.get('source', 'Unknown')}\n"
            f"    Abstract: {abstract[:300]}..."
        )

    works_text = "\n\n".join(works_summary)
    opposing_text = "\n".join([
        f"- {p['position']} (proponents: {', '.join(p['proponents'])})"
        for p in opposing_positions
    ])

    return f"""You are an expert academic research librarian presenting results to a scholar.

The researcher's context:
"{research_context}"

Known opposing scholarly positions:
{opposing_text}

Here are {len(works)} retrieved works. Analyze them and respond in exactly three clearly labeled sections:

---LAYER1---
List only works scoring 6 or higher for relevance. For each include:
- Title, Author, Year
- Why it is relevant to this specific research context
- Its scholarly position or perspective

---LAYER2---
Intellectual Lineage: Which foundational works, earlier scholars, or primary sources shaped the works in Layer 1? Trace the intellectual ancestry of this field as it relates to the research context.

---LAYER3---
Scholarly Debate: What are the core disagreements among scholars on this topic? Which works or scholars represent opposing camps? What should the researcher understand about these tensions?

Works to analyze:
{works_text}

Be direct, scholarly, and honest. If works are not relevant, exclude them from Layer 1."""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    research_context = request.json.get("context", "")

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Identifying key scholars and search terms...'})}\n\n"

            intelligence = identify_key_scholars(research_context)

            yield f"data: {json.dumps({'type': 'scholars', 'data': intelligence})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching academic databases...'})}\n\n"

            all_works = []
            for scholar in intelligence["scholars"]:
                works = search_by_author(scholar)
                all_works.extend(works)
                ss_works = search_semantic_scholar_by_author(scholar)
                all_works.extend(ss_works)

            for term in intelligence["search_terms"]:
                works = search_by_term(term)
                all_works.extend(works)
                ss_works = search_semantic_scholar_by_term(term)
                all_works.extend(ss_works)

            unique_works = deduplicate_works(all_works)
            yield f"data: {json.dumps({'type': 'status', 'message': f'Analyzing {len(unique_works)} works across three layers...'})}\n\n"

            prompt = build_analysis_prompt(
                research_context,
                unique_works[:20],
                intelligence["opposing_positions"]
            )

            full_response = ""
            with client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
