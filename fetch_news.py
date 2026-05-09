import os
import re
import json
import time
import hashlib
import traceback
import feedparser
import requests
from datetime import datetime, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────
NEWS_API_KEY   = os.environ["NEWS_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT  = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_KEY     = os.environ["GEMINI_API_KEY"]

SEEN_FILE      = "seen_hashes.json"
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ── Keyword filters ───────────────────────────────────────────────────────────

# Defense news: needs 2+ hits from this list (non-ArXiv sources)
DEFENSE_KEYWORDS = [
    "military", "weapon", "warfare", "pentagon", "nato", "darpa",
    "battlefield", "combat", "armed forces", "defence", "defense department",
    "lockheed", "raytheon", "northrop", "bae systems", "counter-drone",
    "missile", "national security", "intelligence agency", "cia", "nsa",
    "cyber attack", "cyberwarfare", "signals intelligence", "autonomous weapon",
    "lethal autonomous", "war", "troops", "soldier", "army", "navy", "air force",
]

# ArXiv defense: needs 1 hit from this stricter list
DEFENSE_ARXIV_STRICT = [
    "military", "weapon", "warfare", "combat", "battlefield",
    "missile", "drone strike", "cyber attack", "national security",
]

# Tools for NewsAPI / HN
TOOLS_KEYWORDS = [
    "launch", "releases", "released", "introducing", "new model", "now available",
    "open source", "open-source", "free tier", "just dropped", "available now",
    "GPT", "Claude", "Gemini", "Llama", "Mistral", "Qwen", "Phi", "DeepSeek",
    "Grok", "Hugging Face", "Ollama", "LangChain", "CrewAI", "AutoGen",
    "ComfyUI", "Stable Diffusion", "Flux", "image generation", "code assistant",
    "AI assistant", "plugin", "extension", "API update",
]

# ArXiv only: paper must EXPLICITLY release code/weights/dataset at a URL
ARXIV_RELEASE_KEYWORDS = [
    "github.com", "huggingface.co", "pip install", "model weights",
    "we open-source", "we open source", "code is available", "code available at",
    "released at", "available at https", "dataset available",
]

GENERAL_AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "large language model",
    "generative AI", "AI regulation", "AI safety", "AI policy", "AI funding",
    "AI startup", "AI company", "AI chip", "AI hardware", "AGI",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_seen() -> set:
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
    t = text.lower()
    return sum(1 for k in keywords if k.lower() in t)

def matches_any(text: str, keywords: list) -> bool:
    return count_matches(text, keywords) > 0

def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", text).strip()

def clean_title(title: str, maxlen: int = 80) -> str:
    title = re.split(r" [-|] ", title)[0].strip()
    return title[:maxlen] + "…" if len(title) > maxlen else title

# ── Gemini summariser ─────────────────────────────────────────────────────────

def summarise_batch(items: list[dict]) -> list[dict]:
    """Summarise all items in one Gemini API call. Falls back to raw text on error."""
    if not items:
        return items

    numbered = ""
    for i, item in enumerate(items, 1):
        raw = strip_html(item.get("summary", "") or item.get("title", ""))
        # Strip the ArXiv preamble that always appears
        raw = re.sub(r"arXiv:\S+\s*Announce Type:\s*\w+\s*Abstract:\s*", "", raw).strip()
        numbered += f"{i}. TITLE: {item['title']}\n   TEXT: {raw[:500]}\n\n"

    prompt = (
        "You are summarising AI news for a curious non-expert reader.\n\n"
        "For each numbered item below write EXACTLY 2 short plain-English sentences:\n"
        "- Sentence 1: What happened or what this is (no jargon, no 'the paper proposes').\n"
        "- Sentence 2: Why it matters or what someone can practically do with it.\n\n"
        "Rules:\n"
        "- Write like texting a smart friend, not writing an abstract.\n"
        "- Keep each full summary under 200 characters total.\n"
        "- If it is a pure academic paper with no practical use, write: "
        "'Researchers explored [topic in plain words]. Mainly relevant to AI scientists for now.'\n"
        "- Return ONLY a valid JSON array of strings, one per item, in order.\n"
        "- No markdown fences, no extra text, no trailing commas.\n\n"
        f"Items:\n{numbered}\n"
        'Return format: ["summary 1", "summary 2", ...]'
    )

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_KEY},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 1500,
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                },
            },
            timeout=40,
        )
        print(f"  Gemini status: {resp.status_code}")
        if not resp.ok:
            print(f"  Gemini error body: {resp.text[:500]}")
            resp.raise_for_status()

        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"  Gemini raw response (first 300 chars): {raw_text[:300]}")

        # Strip markdown fences if Gemini wraps them anyway
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        summaries = json.loads(raw_text)

        for i, item in enumerate(items):
            if i < len(summaries):
                item["summary"] = str(summaries[i])
        print(f"  Summarised {len(summaries)} items via Gemini 2.0 Flash.")
        return items

    except Exception as e:
        print(f"  Gemini summarise FAILED: {e}")
        traceback.print_exc()
        # Fallback: strip ArXiv preamble and truncate
        for item in items:
            raw = strip_html(item.get("summary", "") or "")
            raw = re.sub(r"arXiv:\S+\s*Announce Type:\s*\w+\s*Abstract:\s*", "", raw).strip()
            item["summary"] = raw[:200] + "…" if len(raw) > 200 else raw
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
        since_ts = int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp())
        r = requests.get(
            "http://hn.algolia.com/api/v1/search",
            params={
                "query": "AI tool launch release open source LLM model",
                "tags": "story",
                "numericFilters": f"created_at_i>{since_ts}",
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
            for entry in feed.entries[:20]:
                items.append({
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "summary": strip_html(entry.get("summary", "")),
                    "source": "ArXiv",
                })
        except Exception as e:
            print(f"ArXiv error ({feed_url}): {e}")
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

        # ── Defense check ──
        if is_arxiv:
            is_defense = count_matches(blob, DEFENSE_ARXIV_STRICT) >= 1
        else:
            is_defense = count_matches(blob, DEFENSE_KEYWORDS) >= 2

        # ── Tools check ──
        if is_arxiv:
            # ArXiv: must contain an actual release URL or explicit open-source statement
            is_tool = matches_any(blob, ARXIV_RELEASE_KEYWORDS)
        else:
            is_tool = matches_any(blob, TOOLS_KEYWORDS)

        is_ai = matches_any(blob, GENERAL_AI_KEYWORDS)

        if is_defense:
            defense.append(item)
        elif is_arxiv and is_tool:
            tools.append(item)
        elif not is_arxiv and (is_tool or is_ai):
            tools.append(item)

    return defense[:5], tools[:7]

# ── Escape MarkdownV2 ─────────────────────────────────────────────────────────

def esc(text: str) -> str:
    for ch in r"_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text

# ── Format Telegram message ───────────────────────────────────────────────────

def build_message(defense: list[dict], tools: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    lines = [f"🤖 *AI Daily Digest — {esc(today)}*\n"]

    def fmt(items: list[dict]) -> list[str]:
        out = []
        for i, item in enumerate(items, 1):
            title = esc(clean_title(item["title"]))
            summ  = esc((item.get("summary") or "").strip())
            src   = esc(item["source"])
            url   = item["url"]
            out.append(f"*{i}\\. [{title}]({url})*")
            if summ:
                out.append(f"_{summ}_")
            out.append(f"📌 `{src}`\n")
        return out

    lines.append("🛡 *AI IN DEFENSE*")
    lines.extend(fmt(defense) if defense else ["_No major defense stories today\\._\n"])

    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("\n🛠 *NEW AI TOOLS & RELEASES*")
    lines.extend(fmt(tools) if tools else ["_Nothing notable released today\\._\n"])

    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("_Powered by NewsAPI · HN · ArXiv · Gemini_")
    return "\n".join(lines)

# ── Send Telegram ─────────────────────────────────────────────────────────────

def send_telegram(text: str):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "MarkdownV2",
              "disable_web_page_preview": False},
        timeout=15,
    )
    if not r.ok:
        print(f"Telegram MarkdownV2 failed ({r.status_code}): {r.text[:300]}")
        print("Retrying as plain text…")
        plain = re.sub(r"\\([_*\[\]()~`>#+=|{}.!-])", r"\1", text)
        r2 = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": plain[:4000]},
            timeout=15,
        )
        r2.raise_for_status()
    else:
        print("Telegram: message sent.")

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
    print(f"Raw items fetched: {len(all_items)}")

    defense, tools = categorise(all_items, seen)
    print(f"After filter — Defense: {len(defense)}, Tools: {len(tools)}")

    if not defense and not tools:
        send_telegram("🤖 *AI Daily Digest*\n\n_No relevant stories found today\\._")
        save_seen(seen)
        return

    print("Summarising with Gemini…")
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
