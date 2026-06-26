import requests
import os
from google import genai
from google.genai import types

# Initialize Gemini Client
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"

TEST_POST_IDS = [
    "1u2w74e", # "stopping GS1 UPC prefix payments"
    "1tzk421", # "planning product launch on Amazon, considering investing in UPC"
    "1ucu3mx", # "needs barcodes for Amazon, frustrated with GS1 initial/renewal costs"
    "1udh1h8"  # "e-commerce needing GTINs to resolve Google Merchant Center disapprovals"
]

def fetch_post_content(post_id):
    """Fetches the title and selftext of a real Reddit post using its RSS feed."""
    import xml.etree.ElementTree as ET
    import re
    import html as html_lib
    
    url = f"https://www.reddit.com/comments/{post_id}/.rss"
    headers = {'User-Agent': 'python:test_reply_drafter:v1.0 (by /u/Barcodescoder)'}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        
        if entry is None:
            return None
            
        title = entry.find('atom:title', ns).text
        content = entry.find('atom:content', ns)
        
        # Parse content
        body = ""
        if content is not None and content.text:
            text = html_lib.unescape(content.text)
            body = re.sub(r'<[^>]+>', '', text).strip()
            
        # Try to extract subreddit from category or assume from known context
        category = root.find('atom:category', ns)
        subreddit = category.attrib.get('label', 'unknown') if category is not None else 'unknown'
        
        return {
            'title': title,
            'selftext': body,
            'subreddit': subreddit
        }
    except Exception as e:
        print(f"Error fetching post {post_id}: {e}")
        return None

def draft_reply(post):
    """Drafts a casual reply using Gemini."""
    
    system_instruction = """You are monitoring Reddit to help users who are confused or complaining about the high cost of GS1 barcodes. 
Your goal is to provide a very brief, direct, and factual alternative.

Key knowledge:
- GS1 is the largest barcode supplier, but they are expensive with annual fees.
- Resellers provide functional, legal barcodes with NO annual license fees (one-off payment).
- The International Barcodes Network is a trustworthy network of such resellers.

Strict Guidelines for your reply:
- Tone: Extremely brief, direct, and factual. Avoid AI-sounding fluff, fake empathy (e.g., "I totally get it!"), and unnecessary pleasantries.
- Length: 2-3 sentences max. 
- Content: Directly answer their core issue. ALWAYS mention the "International Barcodes Network" as the trustworthy reseller alternative. 
- Value Prop: Highlight that our barcodes have NO annual license fees and are affordable for small businesses.
- Relevance: DO NOT mention Amazon or specific retailers (like Walmart/Macy's). If the post mentions Amazon, do not encourage them to use our barcodes for Amazon, as Amazon now strictly requires GS1 barcodes. Match the region of the user if relevant, avoid US-centric examples for non-US posts.
- Do NOT sound like an AI. No generic intros or outros.

Example of a perfect reply:
GS1 is the largest barcode supplier in the world, but not the only one. You can get functional, legal barcodes from trustworthy barcode resellers such as the International Barcodes Network. These barcodes don't have annual license fees either, so they are much more affordable and practical for a small business than joining GS1."""

    prompt = f"""Please draft a Reddit reply to the following post:

Subreddit: r/{post['subreddit']}
Title: {post['title']}
Body: {post['selftext']}
"""

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        ),
    )
    return response.text

if __name__ == "__main__":
    import time
    print("Starting AI Reply Drafter Test...\n")
    for pid in TEST_POST_IDS:
        print(f"--- Fetching Post ID: {pid} ---")
        post = fetch_post_content(pid)
        if post:
            print(f"Title: {post['title']}")
            print("Drafting reply...\n")
            reply = draft_reply(post)
            print("=== AI DRAFTED REPLY ===")
            print(reply)
            print("========================\n")
        else:
            print("Failed to fetch post or parse content.")
        time.sleep(10)
