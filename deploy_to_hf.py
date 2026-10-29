"""
Hugging Face Spaces Deployment Script
Deploys the Movie Recommender System to Hugging Face Spaces (Streamlit SDK).
Uploads code and model artifacts (via automatic LFS) and configures Space secrets.
"""

import argparse
import getpass
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MOVIE_LIST_PATH = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")


def ensure_models_exist():
    """Ensure precomputed models exist before deployment."""
    if not os.path.exists(MOVIE_LIST_PATH) or not os.path.exists(SIMILARITY_PATH):
        print("\n[!] Precomputed models not found. Building them now...")
        from generate_model import build_model
        build_model(data_dir=os.path.join(BASE_DIR, "data"), model_dir=MODEL_DIR)


def deploy(token=None, space_name="movie-recommender-system", private=False):
    # 1. Resolve Hugging Face Token
    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n=======================================================")
        print("  Hugging Face Deployment Setup")
        print("=======================================================")
        print("To deploy to Hugging Face, you need an Access Token with 'Write' permission.")
        print("Get your free token here: https://huggingface.co/settings/tokens\n")
        try:
            hf_token = getpass.getpass("Enter your Hugging Face Token (input hidden): ").strip()
        except Exception:
            hf_token = input("Enter your Hugging Face Token: ").strip()

    if not hf_token:
        print("[ERROR] Hugging Face token is required for deployment.", file=sys.stderr)
        sys.exit(1)

    print("\n1. Authenticating with Hugging Face...")
    login(token=hf_token, add_to_git_credential=False)
    api = HfApi(token=hf_token)

    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"   Authenticated as: {username} ({user_info.get('email', '')})")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate: {e}", file=sys.stderr)
        sys.exit(1)

    repo_id = f"{username}/{space_name}"
    print(f"\n2. Checking Space target: {repo_id}...")

    # 2. Create Space if it doesn't already exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            private=private,
            exist_ok=True,
        )
        print(f"   Space '{repo_id}' is ready.")
    except Exception as e:
        print(f"[ERROR] Could not create Space: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Configure TMDB API Key Secret if available
    tmdb_key = os.getenv("TMDB_API_KEY", "").strip()
    if tmdb_key and tmdb_key.lower() not in ("your_tmdb_api_key_here", "none", ""):
        print("\n3. Setting TMDB_API_KEY as a Space Secret...")
        try:
            api.add_space_secret(repo_id=repo_id, key="TMDB_API_KEY", value=tmdb_key)
            print("   TMDB_API_KEY secret configured successfully!")
        except Exception as e:
            print(f"   [Notice] Could not set secret automatically: {e}")
            print("   You can add it manually under Space Settings -> Variables and secrets.")
    else:
        print("\n3. TMDB_API_KEY not found in .env. Posters will use fallback placeholders.")

    # 4. Ensure models exist
    ensure_models_exist()

    # 5. Upload files to Space (automatically uses Git LFS for model/*.pkl)
    print(f"\n4. Uploading application files & model artifacts to Hugging Face Space...")
    print("   (similarity.pkl will be uploaded via Git LFS - this may take 1-2 minutes depending on bandwidth)...")

    ignore_patterns = [
        ".venv/**",
        ".git/**",
        ".gitignore",
        "__pycache__/**",
        ".env",
        ".env.example",
        "data/**",
        "*.pyc",
        "scratch/**",
        "deploy_to_hf.py",
    ]

    try:
        api.upload_folder(
            folder_path=BASE_DIR,
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Deploy Movie Recommender System (app + models)",
        )
        print("\n[SUCCESS] Files and models uploaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to upload files: {e}", file=sys.stderr)
        sys.exit(1)

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n=======================================================")
    print(f"🚀 Space deployed successfully!")
    print(f"🔗 Live App URL: {space_url}")
    print("=======================================================")
    print("The Space will build its container and launch in ~1-2 minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Movie Recommender System to Hugging Face Spaces")
    parser.add_argument("--token", help="Hugging Face Write Token (or set HF_TOKEN env var)")
    parser.add_argument("--space-name", default="movie-recommender-system", help="Name of the Hugging Face Space")
    parser.add_argument("--private", action="store_true", help="Create a private space instead of public")
    args = parser.parse_args()

    deploy(token=args.token, space_name=args.space_name, private=args.private)
