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
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
KEYWORDS = [
    'EAN-13', 'EAN barcode', 'UPC-A', 'UPC barcode', 'GTIN', 'GS1',
    'buy barcode', 'buy barcodes', 'get barcodes', 'product barcode',
    'retail barcode', 'barcode for Amazon', 'barcode for Shopify'
]
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
1. Someone wanting to BUY or obtain retail barcodes to sell physical products in shops or on online marketplaces (Amazon, eBay, Shopify, etc.).
2. Someone frustrated with or complaining about GS1 — its pricing, membership fees, renewal costs, or sign-up process.

Do NOT flag:
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
    for attempt in range(2):
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
            time.sleep(2)
            
    # Fallback if both attempts fail
    return True, "AI check failed — manual review"

# --- SLACK NOTIFICATION ---
def send_slack_digest(matches, stats):
    if not matches:
        print("No matches to send.")
        return True

    if not SLACK_WEBHOOK_URL:
        print("\n--- TEST RUN: SLACK_WEBHOOK_URL missing ---")
        print("Would have sent the following matches to Slack:")
        for m in matches:
            print(f" - {m['title']} (r/{m['sub']})")
            print(f"   Reason: {m['reason']}")
            print(f"   Link: {m['link']}")
        print("------------------------------------------\n")
        # In test mode without a webhook, we pretend it succeeded so they get marked as seen
        return True

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
                matches.append(post)
            else:
                # Mark irrelevant posts as seen immediately
                seen[post['id']] = datetime.now(timezone.utc).timestamp()
                
        # Polite throttle
        time.sleep(10)
        
    if matches:
        if send_slack_digest(matches, stats):
            # Only mark relevant posts as seen if Slack succeeded
            for m in matches:
                seen[m['id']] = datetime.now(timezone.utc).timestamp()
        else:
            print("Slack delivery failed. Relevant posts not marked as seen.")
    else:
        print("No new relevant posts found today.")
        
    # Always save seen (which now includes all irrelevant posts from today + successful matches)
    save_seen(seen)
        
    print("Run complete.")

if __name__ == '__main__':
    main()
