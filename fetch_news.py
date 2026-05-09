import os
import re
import json
import time
import hashlib
import feedparser
import requests
from datetime import datetime, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────
NEWS_API_KEY    = os.environ["NEWS_API_KEY"]
TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT   = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_KEY      = os.environ["GEMINI_API_KEY"]

SEEN_FILE = "seen_hashes.json"

# ── Keyword filters ───────────────────────────────────────────────────────────
# Defense: needs 2+ matches to avoid false positives like academic papers
DEFENSE_KEYWORDS = [
    "military", "weapon", "warfare", "pentagon", "nato", "darpa",
    "battlefield", "combat", "armed forces", "defence", "defense department",
    "lockheed", "raytheon", "northrop", "bae systems", "counter-drone",
    "missile", "national security", "intelligence agency", "cia", "nsa",
    "cyber attack", "cyberwarfare", "signals intelligence", "autonomous weapon",
    "lethal autonomous", "war", "troops", "soldier", "army", "navy", "air force",
]

# For ArXiv: only include if these very specific terms appear
DEFENSE_ARXIV_STRICT = [
    "military", "weapon", "warfare", "defense", "defence", "combat",
    "battlefield", "missile", "drone strike", "cyber attack", "national security",
]

TOOLS_KEYWORDS = [
    "launch", "releases", "released", "introducing", "new model", "now available",
    "open source", "open-source", "free tier", "just dropped", "available now",
    "GPT", "Claude", "Gemini", "Llama", "Mistral", "Qwen", "Phi", "DeepSeek",
    "Grok", "Hugging Face", "Ollama", "LangChain", "CrewAI", "AutoGen",
    "ComfyUI", "Stable Diffusion", "Flux", "image generation", "code assistant",
    "AI assistant", "AI agent", "plugin", "extension", "API update",
]

GENERAL_AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "large language model",
    "generative AI", "AI regulation", "AI safety", "AI policy", "AI funding",
    "AI startup", "AI company", "AI chip", "AI hardware", "AGI",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_seen():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()

def save_seen(seen: set):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen)[-600:], f)

def make_hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()[:10]

def count_matches(text: str, keywords: list) -> int:
    text_lower = text.lower()
    return sum(1 for k in keywords if k.lower() in text_lower)

def matches_any(text: str, keywords: list) -> bool:
    return count_matches(text, keywords) > 0

def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", text).strip()

def truncate_title(title: str, maxlen: int = 80) -> str:
    # Remove source suffixes like " - Reuters" or " | TechCrunch"
    title = re.split(r" [-|] ", title)[0].strip()
    return title[:maxlen] + "…" if len(title) > maxlen else title

# ── Gemini summariser ─────────────────────────────────────────────────────────

def summarise_batch(items: list[dict]) -> list[dict]:
    """
    Send all items to Gemini 2.0 Flash in one API call.
    Replaces each item's 'summary' with a plain-English 2-sentence version.
    """
    if not items:
        return items

    numbered = ""
    for i, item in enumerate(items, 1):
        raw = strip_html(item.get("summary", "") or item.get("title", ""))
        numbered += f"{i}. TITLE: {item['title']}\n   TEXT: {raw[:600]}\n\n"

    prompt = f"""You are summarising AI news for a curious non-expert reader.

For each numbered item below write EXACTLY 2 short plain-English sentences:
- Sentence 1: What happened or what this is (no jargon).
- Sentence 2: Why it matters or what someone can do with it.

Rules:
- Write like you are texting a smart friend, not writing a paper.
- Keep each full summary under 180 characters.
- If it is a pure academic paper with no practical use yet, say: "Researchers studied [X in simple words]. Mainly useful for AI scientists right now."
- Return ONLY a JSON array of strings, one per item, in order. No markdown, no extra text.

Items:
{numbered}

Return format example: ["summary for item 1", "summary for item 2"]"""

    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models"
            f"/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
        )
        r = requests.post(
            url,
            headers={"content-type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.3},
            },
            timeout=30,
        )
        r.raise_for_status()
        raw_text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw_text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        summaries = json.loads(raw_text)
        for i, item in enumerate(items):
            if i < len(summaries):
                item["summary"] = summaries[i]
        print(f"  Summarised {len(summaries)} items via Gemini 2.0 Flash.")
        return items
    except Exception as e:
        print(f"Gemini summarise error: {e}")
        # Fallback: use truncated raw text
        for item in items:
            raw = strip_html(item.get("summary", "") or "")
            item["summary"] = raw[:180] + "…" if len(raw) > 180 else raw
        return items

# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_newsapi(query: str, page_size: int = 10) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query, "from": since, "sortBy": "publishedAt",
                "pageSize": page_size, "language": "en", "apiKey": NEWS_API_KEY,
            },
            timeout=15,
        )
        r.raise_for_status()
        return [
            {
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "summary": strip_html(a.get("description") or a.get("content") or ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
            }
            for a in r.json().get("articles", [])
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def fetch_hackernews() -> list[dict]:
    try:
        r = requests.get(
            "http://hn.algolia.com/api/v1/search",
            params={
                "query": "AI tool launch release open source LLM model",
                "tags": "story",
                "numericFilters": f"created_at_i>{int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp())}",
                "hitsPerPage": 20,
            },
            timeout=15,
        )
        r.raise_for_status()
        return [
            {
                "title": h.get("title", ""),
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "summary": "",
                "source": "Hacker News",
            }
            for h in r.json().get("hits", [])
            if h.get("title")
        ]
    except Exception as e:
        print(f"HN error: {e}")
        return []


def fetch_arxiv() -> list[dict]:
    items = []
    for feed_url in ["https://rss.arxiv.org/rss/cs.AI", "https://rss.arxiv.org/rss/cs.LG"]:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                items.append({
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "summary": strip_html(entry.get("summary", "")),
                    "source": "ArXiv",
                })
        except Exception as e:
            print(f"ArXiv error: {e}")
    return items

# ── Categorise ────────────────────────────────────────────────────────────────

def categorise(items: list[dict], seen: set) -> tuple[list, list]:
    defense, tools = [], []

    for item in items:
        if not item.get("title"):
            continue
        h = make_hash(item["title"])
        if h in seen:
            continue
        seen.add(h)

        blob = f"{item['title']} {item.get('summary', '')}"
        is_arxiv = item["source"] == "ArXiv"

        # Defense: strict thresholds
        if is_arxiv:
            is_defense = count_matches(blob, DEFENSE_ARXIV_STRICT) >= 1
        else:
            is_defense = count_matches(blob, DEFENSE_KEYWORDS) >= 2

        is_tool = matches_any(blob, TOOLS_KEYWORDS)
        is_ai   = matches_any(blob, GENERAL_AI_KEYWORDS)

        if is_defense:
            defense.append(item)
        elif is_tool or (is_ai and not is_arxiv):
            # ArXiv papers only qualify if they match tools keywords directly
            if is_arxiv and not is_tool:
                continue
            tools.append(item)

    return defense[:5], tools[:7]

# ── Escape MarkdownV2 ─────────────────────────────────────────────────────────

ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"

def esc(text: str) -> str:
    for ch in ESCAPE_CHARS:
        text = text.replace(ch, f"\\{ch}")
    return text

# ── Format Telegram message ───────────────────────────────────────────────────

def build_message(defense: list[dict], tools: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    lines = [f"🤖 *AI Daily Digest — {esc(today)}*\n"]

    def format_section(items: list[dict]) -> list[str]:
        out = []
        for i, item in enumerate(items, 1):
            title = esc(truncate_title(item["title"]))
            url   = item["url"]
            src   = esc(item["source"])
            summ  = esc(item.get("summary", "").strip())
            out.append(f"*{i}\\. [{title}]({url})*")
            if summ:
                out.append(f"_{summ}_")
            out.append(f"📌 `{src}`\n")
        return out

    lines.append("🛡 *AI IN DEFENSE*")
    if defense:
        lines.extend(format_section(defense))
    else:
        lines.append("_No major defense stories today\\._\n")

    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("\n🛠 *NEW AI TOOLS & RELEASES*")
    if tools:
        lines.extend(format_section(tools))
    else:
        lines.append("_Nothing notable released today\\._\n")

    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("_Powered by NewsAPI · HN · ArXiv · Claude_")
    return "\n".join(lines)

# ── Send to Telegram ──────────────────────────────────────────────────────────

def send_telegram(text: str):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": False,
        },
        timeout=15,
    )
    if not r.ok:
        print(f"Telegram MarkdownV2 error {r.status_code}: {r.text}")
        print("Retrying as plain text…")
        r2 = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": text[:4000]},
            timeout=15,
        )
        r2.raise_for_status()
    else:
        print("Message sent successfully.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seen = load_seen()

    print("Fetching sources…")
    all_items = (
        fetch_newsapi("AI military defense autonomous weapon drone surveillance")
        + fetch_newsapi("AI tool release launch open source LLM model agent")
        + fetch_hackernews()
        + fetch_arxiv()
    )
    print(f"Raw items: {len(all_items)}")

    defense, tools = categorise(all_items, seen)
    print(f"After filter — Defense: {len(defense)}, Tools: {len(tools)}")

    if not defense and not tools:
        send_telegram("🤖 *AI Daily Digest*\n\n_No relevant stories found today\\._")
        save_seen(seen)
        return

    print("Summarising with Claude Haiku…")
    defense = summarise_batch(defense)
    time.sleep(1)
    tools   = summarise_batch(tools)

    message = build_message(defense, tools)
    print("Sending to Telegram…")
    send_telegram(message)
    save_seen(seen)
    print("Done.")

if __name__ == "__main__":
    main()
