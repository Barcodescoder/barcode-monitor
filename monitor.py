import os
import sys
import json
import time
import smtplib
import urllib.parse
from datetime import datetime, timezone, timedelta
from html import unescape
import re

import requests
import feedparser
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
KEYWORDS = [
    'EAN-13', 'EAN barcode', 'UPC-A', 'UPC barcode', 'GTIN', 'GS1',
    'buy barcode', 'buy barcodes', 'get barcodes', 'product barcode',
    'retail barcode', 'barcode for Shopify'
]
EXCLUDED_SUBREDDITS = {
    'vedic_astrology_free', 'vedicastrologyexperts', 'vedicastromedia', 'jyotisha_astro', 
    'free_vedic_astro', 'jyotishh', 'vedicastrologyreal', 'jeeadv27dailyupdates', 
    'medicoretards', 'easportsfc', 'footballmanagergames', 'wiiu', 'victoria3', 
    'mapporn', 'comicbookspeculation', '80s90scomics', 'comicbooks', 'cgccomics', 
    'hotwheels', 'pokeinvesting', 'pkmntcgtrades', 'pokemonraffles', 
    'onepiecetcgfinance', 'vintagetoys', 'lawnmowers', 'smallenginerepair', 
    'ar15', 'fnherstal', 'upsc', 'upscprelims2026', 'upsc_forum', 'mpscprep', 
    'indianacademia', 'neetard', 'sixthgrade', 'splatoon', 'goldensun', 
    'rimworld', 'eu4mods', 'market76', 'flightsim', 'gravelcycling', 
    'ultracycling', 'doordashdrivers', 'amazonflexdrivers', 'amazondspdrivers', 
    'cvs', 'michaelsemployees', 'staples', 'lowes', 'kroger', 'walmart', 
    'awesomefreebies', 'beermoneyuk', 'beermoneyideas', 'sidehustlegold', 
    'frozendinners', 'nosleep', 'suicidewatch', 'cathlablounge', 'askelectricians'
}
FETCH_LIMIT = 30
MAX_RETRIES = 5
PRUNE_DAYS = 30
USER_AGENT = 'barcode-lead-monitor/1.0 (daily RSS reader; contact: support@barcode1.co.uk)'
SEEN_FILE = 'seen.json'
MODEL_ID = 'gemini-2.5-flash'
TRUNCATE_LEN = 1500

# Secrets from environment
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL', '')

# --- MODELS ---
class RelevanceVerdict(BaseModel):
    relevant: bool = Field(description="Whether the post is relevant to selling retail barcodes.")
    reason: str = Field(description="Max 20 words explaining the decision.")

# --- HELPERS ---
def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading seen.json: {e}")
    return {}

def save_seen(seen):
    # Prune old entries
    cutoff = (datetime.now(timezone.utc) - timedelta(days=PRUNE_DAYS)).timestamp()
    pruned = {k: v for k, v in seen.items() if v > cutoff}
    with open(SEEN_FILE, 'w') as f:
        json.dump(pruned, f, indent=2)

def strip_tags(html):
    if not html:
        return ""
    text = unescape(html)
    return re.sub(r'<[^>]+>', '', text).strip()

def extract_post_id(permalink):
    # E.g., https://www.reddit.com/r/EtsySellers/comments/12345/title/
    match = re.search(r'/comments/([^/]+)/', permalink)
    return match.group(1) if match else None

# --- FETCH LAYER ---
def fetch_posts_for_keyword(keyword):
    encoded = urllib.parse.quote(keyword)
    endpoints = [
        f"https://www.reddit.com/search/.rss?q={encoded}&sort=new&limit={FETCH_LIMIT}",
        f"https://old.reddit.com/search/.rss?q={encoded}&sort=new&limit={FETCH_LIMIT}",
        f"https://www.reddit.com/search.json?q={encoded}&sort=new&limit={FETCH_LIMIT}"
    ]
    
    headers = {'User-Agent': USER_AGENT}
    
    for attempt in range(MAX_RETRIES):
        for endpoint in endpoints:
            try:
                print(f"Fetching {endpoint} (Attempt {attempt+1})")
                resp = requests.get(endpoint, headers=headers, timeout=15)
                if resp.status_code == 200:
                    if '.json' in endpoint:
                        return parse_json_response(resp.json(), keyword)
                    else:
                        return parse_rss_response(resp.content, keyword)
                else:
                    print(f"  HTTP {resp.status_code}")
            except Exception as e:
                print(f"  Error fetching {endpoint}: {e}")
                
        # If all endpoints fail in this attempt, backoff
        backoff = [5, 15, 30, 60, 120]
        sleep_time = backoff[attempt] if attempt < len(backoff) else 120
        print(f"All endpoints failed for '{keyword}', sleeping {sleep_time}s...")
        time.sleep(sleep_time)
        
    print(f"Failed to fetch any data for keyword: '{keyword}' after {MAX_RETRIES} retries.")
    return None

def parse_rss_response(xml_content, keyword):
    d = feedparser.parse(xml_content)
    posts = []
    for entry in d.entries:
        link = entry.get('link', '')
        if '/comments/' not in link:
            continue
            
        post_id = extract_post_id(link)
        if not post_id:
            continue
            
        content = ""
        if 'content' in entry and entry.content:
            content = entry.content[0].value
        elif 'summary' in entry:
            content = entry.summary
            
        posts.append({
            'id': post_id,
            'title': entry.get('title', ''),
            'link': link,
            'author': entry.get('author', '').replace('/u/', ''),
            'sub': re.search(r'reddit\.com/r/([^/]+)', link).group(1) if re.search(r'reddit\.com/r/([^/]+)', link) else '',
            'published': entry.get('published', ''),
            'snippet': strip_tags(content)[:TRUNCATE_LEN],
            'keyword': keyword
        })
    return posts

def parse_json_response(data, keyword):
    posts = []
    try:
        children = data.get('data', {}).get('children', [])
        for child in children:
            item = child.get('data', {})
            link = "https://www.reddit.com" + item.get('permalink', '')
            if '/comments/' not in link:
                continue
                
            post_id = item.get('id', '')
            if not post_id:
                continue
                
            posts.append({
                'id': post_id,
                'title': item.get('title', ''),
                'link': link,
                'author': item.get('author', ''),
                'sub': item.get('subreddit', ''),
                'published': str(item.get('created_utc', '')),
                'snippet': item.get('selftext', '')[:TRUNCATE_LEN],
                'keyword': keyword
            })
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    return posts

# --- AI FILTERING ---
def filter_with_gemini(post, client):
    prompt = f"""You are screening Reddit posts for a company that SELLS retail barcodes (EAN-13, UPC-A, UPC-E, GTIN) to businesses.

Flag a post as relevant ONLY if it shows one of these:
1. Someone wanting to BUY or obtain retail barcodes to sell physical products in shops or on online marketplaces (eBay, Shopify, etc.).
2. Someone frustrated with or complaining about GS1 — its pricing, membership fees, renewal costs, or sign-up process.

Do NOT flag:
- ANY posts where the user is specifically trying to list on Amazon or wants help with Amazon barcode requirements. Amazon recently changed their policy to require GS1 barcodes, so we cannot help them.
- Questions about scanning, reading, or decoding barcodes
- Generating or printing barcodes in software/code
- ISBN, FNSKU, QR codes, or Data Matrix questions
- Inventory or POS software how-tos
- Hobby or non-commercial projects
- Anything that is not a prospective barcode buyer or a GS1-frustrated seller

Post Title: {post['title']}
Subreddit: r/{post['sub']}
Content: {post['snippet']}
"""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": RelevanceVerdict,
                    "temperature": 0.0
                }
            )
            data = json.loads(response.text)
            return data.get('relevant', False), data.get('reason', '')
        except Exception as e:
            print(f"Gemini API error (Attempt {attempt+1}): {e}")
            time.sleep(5 + (attempt * 5)) # 5, 10, 15, 20, 25 seconds
            
    # Fallback if all attempts fail
    return True, "AI check failed — manual review"

# --- AI DRAFTING ---
def draft_reply(post, client):
    system_instruction = """You are monitoring Reddit to help users who need retail barcodes (UPC/EAN) or are frustrated by the high cost of GS1 barcodes. 
Your goal is to provide a very brief, direct, and factual alternative to GS1.

Key knowledge:
- GS1 is the largest barcode supplier, but they are expensive with annual fees.
- Resellers provide functional, legal barcodes with NO annual license fees (one-off payment).
- The International Barcodes Network is a trustworthy network of such resellers.

Strict Guidelines for your reply:
- Tone: Extremely brief, direct, and factual. Avoid AI-sounding fluff, fake empathy (e.g., "I totally get it!"), and unnecessary pleasantries.
- Length: 2-3 sentences max. 
- Content: Directly answer their core issue. ALWAYS mention the "International Barcodes Network" as the trustworthy reseller alternative. If they just need barcodes to list products, tell them they can get them easily from resellers. DO NOT refuse to answer just because they aren't complaining about GS1.
- Value Prop: Highlight that our barcodes have NO annual license fees and are affordable for small businesses.
- Relevance: DO NOT mention Amazon or specific retailers (like Walmart/Macy's). If the post mentions Amazon, do not encourage them to use our barcodes for Amazon, as Amazon now strictly requires GS1 barcodes. Match the region of the user if relevant, avoid US-centric examples for non-US posts.
- Do NOT sound like an AI. No generic intros or outros.

Example of a perfect reply:
GS1 is the largest barcode supplier in the world, but not the only one. You can get functional, legal barcodes from trustworthy barcode resellers such as the International Barcodes Network. These barcodes don't have annual license fees either, so they are much more affordable and practical for a small business than joining GS1."""

    prompt = f"""Please draft a Reddit reply to the following post:

Subreddit: r/{post['sub']}
Title: {post['title']}
Body: {post.get('selftext', post['snippet'])}
"""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                ),
            )
            return response.text
        except Exception as e:
            print(f"Gemini draft error (Attempt {attempt+1}): {e}")
            time.sleep(5 + (attempt * 5))
    return "Failed to draft reply."

# --- SLACK NOTIFICATION ---
def send_slack_digest(matches, stats):
    if not matches:
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"✅ Barcode Lead Monitor — Run Complete ({datetime.now().strftime('%d %b')})"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "No new relevant posts found today."
                }
            }
        ]
    else:
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 Barcode Leads — {len(matches)} new ({datetime.now().strftime('%d %b')})"
                }
            },
            {"type": "divider"}
        ]

        for m in matches:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{m['title']}*\nr/{m['sub']} | by {m['author']} | {m['published']}\n> *AI Reason:* {m['reason']}\n<{m['link']}|View Post>"
                }
            })
            if 'drafted_reply' in m and m['drafted_reply']:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"🤖 *Drafted Reply:*\n>{m['drafted_reply'].replace(chr(10), chr(10) + '>')}"
                    }
                })
            blocks.append({"type": "divider"})

    failed_text = f" | *Failed Keywords:* {len(stats['failed_keywords'])}" if stats['failed_keywords'] else ""
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Run Stats: *{stats['keywords_checked']}* keywords checked | *{stats['total_fetched']}* posts fetched | *{len(matches)}* matches{failed_text}"
            }
        ]
    })

    if not SLACK_WEBHOOK_URL:
        print("\n--- TEST RUN: SLACK_WEBHOOK_URL missing ---")
        print("Would have sent the following blocks to Slack:")
        import json
        print(json.dumps(blocks, indent=2))
        print("------------------------------------------\n")
        # In test mode without a webhook, we pretend it succeeded so they get marked as seen
        return True

    payload = {"blocks": blocks}

    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code == 200 or resp.status_code == 201:
            print("Slack message sent successfully.")
            return True
        else:
            print(f"Failed to send Slack message. HTTP {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"Error sending to Slack: {e}")
        return False

# --- MAIN RUNNER ---
def main():
    print("Starting daily Reddit Barcode Monitor...")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set.")
        sys.exit(1)
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    seen = load_seen()
    
    stats = {
        'keywords_checked': len(KEYWORDS),
        'total_fetched': 0,
        'failed_keywords': []
    }
    matches = []
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"[{i+1}/{len(KEYWORDS)}] Processing '{keyword}'...")
        posts = fetch_posts_for_keyword(keyword)
        
        if posts is None:
            stats['failed_keywords'].append(keyword)
            continue
            
        posts = [p for p in posts if p['sub'].lower() not in EXCLUDED_SUBREDDITS]
            
        stats['total_fetched'] += len(posts)
        
        print(f"  Fetched {len(posts)} posts. Breakdown:")
        for p in posts:
            print(f"    - ID: {p['id']} | r/{p['sub']} | {p['title'][:60]}")
            
        for post in posts:
            if post['id'] in seen or any(m['id'] == post['id'] for m in matches):
                continue
                
            # Filter
            relevant, reason = filter_with_gemini(post, client)
            post['reason'] = reason
            
            print(f"  -> {post['id']} | Verdict: {'PASS' if relevant else 'FAIL'} | {reason}")
            
            if relevant:
                print("    Drafting reply...")
                post['drafted_reply'] = draft_reply(post, client)
                matches.append(post)
            else:
                # Mark irrelevant posts as seen immediately
                seen[post['id']] = datetime.now(timezone.utc).timestamp()
                
        # Polite throttle
        time.sleep(10)
        
    slack_success = send_slack_digest(matches, stats)
    
    if matches and slack_success:
        # Only mark relevant posts as seen if Slack succeeded
        for m in matches:
            seen[m['id']] = datetime.now(timezone.utc).timestamp()
    elif matches and not slack_success:
        print("Slack delivery failed. Relevant posts not marked as seen.")
    elif not matches:
        print("No new relevant posts found today.")
        
    # Always save seen (which now includes all irrelevant posts from today + successful matches)
    save_seen(seen)
        
    print("Run complete.")

if __name__ == '__main__':
    main()
