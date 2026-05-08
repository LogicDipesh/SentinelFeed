import os
import re
import json
import hashlib
import feedparser
import requests
from datetime import datetime, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────
NEWS_API_KEY   = os.environ["NEWS_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT  = os.environ["TELEGRAM_CHAT_ID"]

SEEN_FILE = "seen_hashes.json"   # persisted via GitHub Actions cache

# ── Keyword filters ───────────────────────────────────────────────────────────
DEFENSE_KEYWORDS = [
    "defense", "defence", "military", "weapon", "drone", "autonomous weapon",
    "pentagon", "nato", "warfare", "surveillance", "cyber attack", "intelligence agency",
    "DARPA", "lockheed", "raytheon", "battlefield", "autonomous system", "counter-drone",
    "missile", "radar", "satellite", "geopolitics", "national security",
]

TOOLS_KEYWORDS = [
    "launch", "release", "introduce", "new model", "open source", "open-source",
    "API", "tool", "framework", "SDK", "free tier", "available now", "just released",
    "GPT", "Claude", "Gemini", "Llama", "Mistral", "Qwen", "Phi", "DeepSeek",
    "Hugging Face", "Ollama", "LangChain", "CrewAI", "AutoGen", "ComfyUI",
    "Stable Diffusion", "image generation", "code generation", "agent",
]

GENERAL_AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning", "neural network",
    "large language model", "LLM", "generative AI", "AI model", "AI system",
    "transformer", "computer vision", "reinforcement learning", "AI research",
    "AI regulation", "AI safety", "AGI", "AI startup", "AI funding",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_seen():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()

def save_seen(seen):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen)[-500:], f)  # keep last 500 to avoid bloat

def make_hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:10]

def matches(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(k.lower() in text_lower for k in keywords)

def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200] + "…" if len(text) > 200 else text

def truncate_title(title: str, maxlen: int = 90) -> str:
    return title[:maxlen] + "…" if len(title) > maxlen else title

# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_newsapi(query: str, page_size: int = 10) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": since,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "language": "en",
        "apiKey": NEWS_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "summary": clean(a.get("description") or a.get("content") or ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def fetch_hackernews() -> list[dict]:
    url = "http://hn.algolia.com/api/v1/search"
    params = {
        "query": "AI tool launch release open source LLM",
        "tags": "story",
        "numericFilters": f"created_at_i>{int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp())}",
        "hitsPerPage": 15,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        return [
            {
                "title": h.get("title", ""),
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "summary": "",
                "source": "Hacker News",
            }
            for h in hits
            if h.get("title")
        ]
    except Exception as e:
        print(f"HN error: {e}")
        return []


def fetch_arxiv() -> list[dict]:
    feeds = [
        "https://rss.arxiv.org/rss/cs.AI",
        "https://rss.arxiv.org/rss/cs.LG",
    ]
    items = []
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                items.append({
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "summary": clean(entry.get("summary", "")),
                    "source": "ArXiv",
                })
        except Exception as e:
            print(f"ArXiv error ({feed_url}): {e}")
    return items

# ── Categorise ────────────────────────────────────────────────────────────────

def categorise(items: list[dict], seen: set) -> tuple[list, list]:
    defense, tools = [], []
    for item in items:
        blob = f"{item['title']} {item['summary']}"
        h = make_hash(item["title"])
        if h in seen:
            continue
        seen.add(h)

        is_defense = matches(blob, DEFENSE_KEYWORDS)
        is_tool    = matches(blob, TOOLS_KEYWORDS)
        is_ai      = matches(blob, GENERAL_AI_KEYWORDS)

        if not (is_defense or is_tool or is_ai):
            continue

        if is_defense:
            defense.append(item)
        elif is_tool or is_ai:
            tools.append(item)

    return defense[:6], tools[:8]

# ── Format Telegram message ───────────────────────────────────────────────────

def build_message(defense: list[dict], tools: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    lines = [f"🤖 *AI Daily Digest — {today}*\n"]

    if defense:
        lines.append("🛡 *AI IN DEFENSE*")
        for i, item in enumerate(defense, 1):
            title = truncate_title(item["title"])
            src   = item["source"]
            url   = item["url"]
            summ  = item["summary"]
            lines.append(f"{i}\\. [{title}]({url})")
            if summ:
                lines.append(f"   _{summ}_")
            lines.append(f"   `{src}`")
            lines.append("")
    else:
        lines.append("🛡 *AI IN DEFENSE*\nNo major stories today\\.\n")

    lines.append("─────────────────────")

    if tools:
        lines.append("\n🛠 *NEW AI TOOLS & RELEASES*")
        for i, item in enumerate(tools, 1):
            title = truncate_title(item["title"])
            src   = item["source"]
            url   = item["url"]
            summ  = item["summary"]
            lines.append(f"{i}\\. [{title}]({url})")
            if summ:
                lines.append(f"   _{summ}_")
            lines.append(f"   `{src}`")
            lines.append("")
    else:
        lines.append("\n🛠 *NEW AI TOOLS & RELEASES*\nNothing notable today\\.\n")

    lines.append("─────────────────────")
    lines.append("_Powered by NewsAPI · HN · ArXiv_")
    return "\n".join(lines)

# ── Send to Telegram ──────────────────────────────────────────────────────────

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT,
        "text": text,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": False,
    }
    r = requests.post(url, json=payload, timeout=15)
    if not r.ok:
        print(f"Telegram error: {r.status_code} {r.text}")
        r.raise_for_status()
    else:
        print("Message sent successfully.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seen = load_seen()

    print("Fetching NewsAPI (defense)…")
    news_defense = fetch_newsapi("AI military defense autonomous weapon drone")
    print("Fetching NewsAPI (tools)…")
    news_tools   = fetch_newsapi("AI tool launch release open source LLM model")
    print("Fetching Hacker News…")
    hn_items     = fetch_hackernews()
    print("Fetching ArXiv…")
    arxiv_items  = fetch_arxiv()

    all_items = news_defense + news_tools + hn_items + arxiv_items
    print(f"Total raw items: {len(all_items)}")

    defense, tools = categorise(all_items, seen)
    print(f"Defense: {len(defense)}, Tools: {len(tools)}")

    message = build_message(defense, tools)
    send_telegram(message)
    save_seen(seen)

if __name__ == "__main__":
    main()
