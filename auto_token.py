#!/usr/bin/env python3
"""
auto_token.py — Zerodha Kite Access Token Auto-Updater
=======================================================
Run ONCE each morning before 9:30 AM IST on trading days.

What this script does automatically:
  1. Opens Zerodha login page in your browser
  2. You log in → paste the redirect URL back here
  3. Exchanges request_token for today's access_token via Kite API
  4. Encrypts the token using GitHub's libsodium public key (PyNaCl)
  5. Updates the KITE_ACCESS_TOKEN GitHub Actions secret automatically

No more manual copy-paste into GitHub! Just run this script and you're done.

First-time setup:
  1. pip install kiteconnect PyNaCl python-dotenv requests
  2. Create a .env file in this folder with:
       KITE_API_KEY=your_api_key
       KITE_API_SECRET=your_api_secret
       GITHUB_PAT=ghp_xxxx...
       GITHUB_REPO=PrinceM21/indian-stock-signals
  3. Create a GitHub fine-grained PAT at:
       https://github.com/settings/personal-access-tokens/new
       → Repository: indian-stock-signals
       → Permission: Secrets → Read and write

Daily usage:
  python auto_token.py
"""

import os
import sys
import base64
import webbrowser
import requests
from urllib.parse import urlparse, parse_qs

# Load .env file if present (silently skip if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # user can also set env vars manually


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Get Kite access token via browser OAuth
# ─────────────────────────────────────────────────────────────────────────────

def get_kite_access_token(api_key, api_secret):
    """
    Interactive Kite OAuth flow:
    Opens browser → user logs in → user pastes redirect URL back →
    script exchanges request_token for access_token.
    Returns the access_token string.
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("ERROR: kiteconnect not installed. Run: pip install kiteconnect==5.0.1")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print(f"\n🌐 Opening Zerodha login page...")
    print(f"   URL: {login_url}\n")
    webbrowser.open(login_url)

    print("After logging in, your browser redirects to a URL like:")
    print("  http://127.0.0.1/?request_token=XXXXXXXXXXXXXXXX&action=login&status=success")
    print("The page will say 'This site can't be reached' — that is NORMAL.\n")

    redirect_url = input("Paste the full redirect URL here:\n> ").strip()

    # Extract request_token from URL
    request_token = None
    try:
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        if "request_token" in params:
            request_token = params["request_token"][0]
        else:
            # Fallback: manual split
            for part in redirect_url.split("&"):
                if "request_token=" in part:
                    request_token = part.split("=", 1)[1]
                    break
    except Exception as e:
        print(f"ERROR parsing URL: {e}")
        sys.exit(1)

    if not request_token:
        print("ERROR: Could not find request_token in the URL.")
        print("Make sure you copied the FULL URL from the browser address bar.")
        sys.exit(1)

    print(f"\n✅ Found request_token: {request_token[:8]}...{request_token[-4:]}")

    # Exchange request_token for access_token
    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]
    except Exception as e:
        print(f"\n❌ ERROR generating session: {e}")
        print("Common causes:")
        print("  - Wrong API Secret")
        print("  - request_token already used (it can only be used once)")
        print("  - request_token expired (must be used within a few minutes of login)")
        sys.exit(1)

    # Quick validation
    try:
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"✅ Logged in as: {profile.get('user_name', 'Unknown')} ({profile.get('email', '')})")
    except Exception as e:
        print(f"⚠️  Token obtained but validation skipped: {e}")

    return access_token


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Get GitHub repo's libsodium public key (needed for encryption)
# ─────────────────────────────────────────────────────────────────────────────

def get_github_repo_public_key(repo, pat):
    """
    Returns (key_id, key_value_base64) from GitHub's secrets encryption endpoint.
    GitHub requires secrets to be encrypted with this key before uploading.
    """
    url = f"https://api.github.com/repos/{repo}/actions/secrets/public-key"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.HTTPError as e:
        if resp.status_code == 401:
            print("ERROR: GitHub PAT is invalid or expired. Create a new one at:")
            print("  https://github.com/settings/personal-access-tokens/new")
        elif resp.status_code == 403:
            print("ERROR: GitHub PAT does not have 'Secrets: Read and write' permission.")
        elif resp.status_code == 404:
            print(f"ERROR: Repository '{repo}' not found. Check GITHUB_REPO in your .env")
        else:
            print(f"ERROR: GitHub API returned {resp.status_code}: {resp.text}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR connecting to GitHub API: {e}")
        sys.exit(1)

    data = resp.json()
    return data["key_id"], data["key"]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Encrypt the secret using PyNaCl (libsodium SealedBox)
# ─────────────────────────────────────────────────────────────────────────────

def encrypt_secret_for_github(public_key_b64, secret_value):
    """
    GitHub requires secrets to be encrypted with the repo's libsodium public key.
    Uses PyNaCl's SealedBox (X25519 + XSalsa20-Poly1305).
    Returns base64-encoded encrypted bytes ready for the GitHub API.
    """
    try:
        from nacl import encoding, public
    except ImportError:
        print("ERROR: PyNaCl not installed. Run: pip install PyNaCl==1.5.0")
        sys.exit(1)

    public_key_bytes = base64.b64decode(public_key_b64)
    public_key = public.PublicKey(public_key_bytes)
    sealed_box = public.SealedBox(public_key)
    encrypted  = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Update the GitHub Actions secret via REST API
# ─────────────────────────────────────────────────────────────────────────────

def update_github_secret(repo, pat, secret_name, secret_value):
    """
    Creates or updates a GitHub Actions secret in the given repo.
    GitHub requires the secret value to be encrypted with the repo's public key.
    """
    print(f"\n🔐 Encrypting token for GitHub...")
    key_id, public_key_b64 = get_github_repo_public_key(repo, pat)
    encrypted_value = encrypt_secret_for_github(public_key_b64, secret_value)

    url = f"https://api.github.com/repos/{repo}/actions/secrets/{secret_name}"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "encrypted_value": encrypted_value,
        "key_id":          key_id,
    }

    try:
        resp = requests.put(url, headers=headers, json=payload, timeout=15)
    except Exception as e:
        print(f"ERROR: Could not connect to GitHub API: {e}")
        sys.exit(1)

    if resp.status_code in (201, 204):
        print(f"✅ GitHub secret '{secret_name}' updated successfully!")
    else:
        print(f"❌ ERROR updating secret: HTTP {resp.status_code}")
        print(f"   Response: {resp.text}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Zerodha Kite → GitHub Secret Auto-Updater")
    print("=" * 60)

    # Load credentials from environment / .env
    api_key     = os.environ.get("KITE_API_KEY",    "").strip()
    api_secret  = os.environ.get("KITE_API_SECRET", "").strip()
    github_pat  = os.environ.get("GITHUB_PAT",      "").strip()
    github_repo = os.environ.get("GITHUB_REPO",     "").strip()

    # Prompt for anything missing
    if not api_key:
        api_key = input("Enter Kite API Key: ").strip()
    if not api_secret:
        api_secret = input("Enter Kite API Secret: ").strip()
    if not github_pat:
        print("\nYou need a GitHub PAT with 'Secrets: Read and write' permission.")
        print("Create one at: https://github.com/settings/personal-access-tokens/new")
        github_pat = input("Enter GitHub PAT: ").strip()
    if not github_repo:
        github_repo = input("Enter GitHub repo (e.g. PrinceM21/indian-stock-signals): ").strip()

    if not all([api_key, api_secret, github_pat, github_repo]):
        print("\nERROR: All four credentials are required.")
        print("Tip: Create a .env file with KITE_API_KEY, KITE_API_SECRET, GITHUB_PAT, GITHUB_REPO")
        sys.exit(1)

    # Step 1: Get Kite access token
    print("\n=== Step 1: Obtain Kite access token ===")
    access_token = get_kite_access_token(api_key, api_secret)

    # Step 2: Update GitHub secret
    print("\n=== Step 2: Update GitHub secret KITE_ACCESS_TOKEN ===")
    update_github_secret(github_repo, github_pat, "KITE_ACCESS_TOKEN", access_token)

    # Done!
    print("\n" + "=" * 60)
    print("✅ All done! GitHub Actions will use the new token.")
    print("⚠️  Token expires at 6:00 AM IST tomorrow.")
    print("   Run this script again tomorrow morning before 9:30 AM IST.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)
