import os, json, glob, requests
from groq import Groq
from concurrent.futures import ThreadPoolExecutor

client = Groq(api_key="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")

SYSTEM_PROMPT = """You are a rigorous biological data extractor and taxonomist with deep expertise in zoology, ecology, and conservation biology.

Your task is to extract or estimate species trait data using the following strict priority order:
1. WIKIPEDIA TEXT (provided) — extract directly if mentioned. This is the highest priority source.
2. IUCN DATA (provided as JSON) — use assessment fields, references, and supplementary info.
3. PEER-REVIEWED KNOWLEDGE — only if above sources are silent; cite the source in parentheses.
4. If completely unknown across all sources, output exactly: "Unknown"

RULES:
- Be maximally precise. Use numeric ranges when available (e.g., "3–7 years", "1.2–2.4 kg").
- Never hallucinate. Every estimate must be scientifically defensible.
- For human_risk_level, choose strictly one of: "Very High", "High", "Caution", "Low".
- For fun facts, prioritize unusual, counterintuitive, or ecologically significant facts.
- short_description must be 1 concise sentence covering taxonomy, habitat, and conservation status.
- Output strictly valid JSON, no extra text."""

def wiki_extract(name):
    try:
        s = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "opensearch", "format": "json", "search": name, "limit": 1
        }, headers={"User-Agent": "Bot/1.0"}, timeout=10).json()
        if not s[1]: return "Not found"
        d = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "titles": s[1][0], "prop": "extracts", "explaintext": True
        }, headers={"User-Agent": "Bot/1.0"}, timeout=10).json()
        return next(iter(d["query"]["pages"].values())).get("extract") or "Not found"
    except Exception: return "Not found"

def inaturalist_photo(name):
    try:
        results = requests.get("https://api.inaturalist.org/v1/taxa",
                               params={"q": name}, timeout=10).json().get("results") or []
        if not results: return None, None
        photo = results[0].get("default_photo") or {}
        return photo.get("medium_url"), photo.get("attribution")
    except Exception: return None, None

def safe_get(d, *keys, default=None):
    for k in keys:
        try: d = d[k]
        except (KeyError, TypeError, IndexError): return default
    return d or default

def process(file):
    try:
        d = json.load(open(file))
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")
        return None

    taxon = d.get("taxon") or {}
    common_names = taxon.get("common_names") or []
    names = [c["name"] for c in common_names if c.get("main")]
    search_name = (names[0] if names else taxon.get("scientific_name", "")).replace(" ", "_")

    photo_url, photo_credit = inaturalist_photo(taxon.get("scientific_name", search_name))

    payload = {
        "assessment_id"  : d.get("assessment_id"),
        "year_published" : d.get("year_published"),
        "scientific_name": taxon.get("scientific_name"),
        "category"       : safe_get(d, "red_list_category", "description", "en"),
        "references"     : d.get("references") or [],
        "url"            : d.get("url"),
        "sis_taxon_id"   : d.get("sis_taxon_id"),
        "photo_url"      : photo_url or "Not available",
        "photo_credit"   : photo_credit or "Not available",
        "wiki_extract"   : wiki_extract(search_name)
    }

    try:
        traits = json.loads(client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract traits.\nIUCN DATA: {payload}\nOUTPUT FORMAT (strict JSON):\n{{'lifespan_years','mass','length','short_description','human_risk_level','human_threat_level','fun_fact_1','fun_fact_2','fun_fact_3'}}"}
            ],
            response_format={"type": "json_object"}
        ).choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM failed for {taxon.get('scientific_name')}: {e}")
        traits = {}

    print(f"✅ {taxon.get('scientific_name')} | 📸 {photo_credit or 'No photo'}")
    return {**payload, **traits}

files = glob.glob("*.txt")
with ThreadPoolExecutor() as ex:
    results = [r for r in ex.map(process, files) if r is not None]  # filter failed files

json.dump(results, open("species_results.json", "w"), indent=2)
print(f"Saved {len(results)} species.")