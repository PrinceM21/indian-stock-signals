#!/usr/bin/env python3
"""
Zerodha Kite Connect — Daily Access Token Generator
====================================================
Run this script ONCE each morning before 9:30 AM IST
to get your daily access_token.

Usage:
  1. Set your API_KEY and API_SECRET below (or use env vars)
  2. Run:  python get_token.py
  3. A browser window will open — log in with your Zerodha credentials
  4. After login, you'll be redirected to http://127.0.0.1/?request_token=XXXX
     (the page will show "This site can't be reached" — that's NORMAL)
  5. Copy the full URL from your browser address bar and paste it here
  6. The script prints your access_token
  7. Go to GitHub → indian-stock-signals → Settings → Secrets → KITE_ACCESS_TOKEN
     and paste the token there

Token expires at 6:00 AM IST the next day — run this script daily on trading days.
"""

import os
import sys
import webbrowser

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Option A: hardcode here (don't commit this file to GitHub if you do this!)
API_KEY    = os.environ.get("KITE_API_KEY",    "")   # or paste your key here
API_SECRET = os.environ.get("KITE_API_SECRET", "")   # or paste your secret here
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Check kiteconnect is installed
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("ERROR: kiteconnect not installed.")
        print("Run:  pip install kiteconnect==5.0.1")
        sys.exit(1)

    # Prompt for credentials if not set
    api_key = API_KEY.strip()
    api_secret = API_SECRET.strip()

    if not api_key:
        api_key = input("Enter your Kite API Key: ").strip()
    if not api_secret:
        api_secret = input("Enter your Kite API Secret: ").strip()

    if not api_key or not api_secret:
        print("ERROR: API key and secret are required.")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)

    # Step 1: Open login URL in browser
    login_url = kite.login_url()
    print(f"\n🌐 Opening Zerodha login page...")
    print(f"   URL: {login_url}\n")
    webbrowser.open(login_url)

    print("After logging in, your browser will go to a URL like:")
    print("  http://127.0.0.1/?request_token=XXXXXXXXXXXXXXXX&action=login&status=success")
    print("The page will say 'This site can't be reached' — that is NORMAL.\n")

    # Step 2: Get request_token from user
    redirected_url = input("Paste the full URL from your browser address bar:\n> ").strip()

    # Extract request_token from URL
    request_token = None
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(redirected_url)
        params = parse_qs(parsed.query)
        if "request_token" in params:
            request_token = params["request_token"][0]
        else:
            # Try to extract manually
            for part in redirected_url.split("&"):
                if part.startswith("request_token=") or "?request_token=" in part:
                    request_token = part.split("=", 1)[1]
                    break
    except Exception as e:
        print(f"ERROR parsing URL: {e}")
        sys.exit(1)

    if not request_token:
        print("ERROR: Could not find request_token in the URL.")
        print("Make sure you copied the full URL from the browser address bar.")
        sys.exit(1)

    print(f"\n✅ Found request_token: {request_token[:8]}...{request_token[-4:]}")

    # Step 3: Exchange for access_token
    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]
    except Exception as e:
        print(f"\n❌ ERROR generating session: {e}")
        print("Common causes:")
        print("  - Wrong API Secret")
        print("  - request_token already used (it can only be used once)")
        print("  - request_token expired (must be used within a few minutes)")
        sys.exit(1)

    # Step 4: Show result
    print("\n" + "="*60)
    print("✅ SUCCESS! Your access token for today:")
    print()
    print(f"  {access_token}")
    print()
    print("="*60)
    print("\nNext steps:")
    print("1. Go to: https://github.com/PrinceM21/indian-stock-signals/settings/secrets/actions")
    print("2. Click 'KITE_ACCESS_TOKEN' → Update secret")
    print("3. Paste the token above → Save")
    print()
    print("⚠️  This token expires at 6:00 AM IST tomorrow.")
    print("   Run this script again tomorrow morning before 9:30 AM IST.")
    print()

    # Optional: quick validation test
    try:
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"🎉 Token verified! Logged in as: {profile.get('user_name', 'Unknown')}")
        print(f"   Broker: {profile.get('broker', 'Zerodha')}")
        print(f"   Email: {profile.get('email', 'N/A')}")
    except Exception as e:
        print(f"(Token validation skipped: {e})")


if __name__ == "__main__":
    main()
