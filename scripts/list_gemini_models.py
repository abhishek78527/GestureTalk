#!/usr/bin/env python3
"""
List models available to a Google Generative Language API key.
Reads GEMINI_API_KEY from env or from .streamlit/secrets.toml.
"""
import os
import sys
import json
import urllib.request
import urllib.error
try:
    import tomllib  # Python 3.11
except Exception:
    tomllib = None

def get_api_key():
    # 1) check env
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # 2) check .streamlit/secrets.toml
    secrets_path = os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml')
    secrets_path = os.path.normpath(secrets_path)
    if os.path.exists(secrets_path) and tomllib is not None:
        with open(secrets_path, 'rb') as f:
            data = tomllib.load(f)
            key = data.get('GEMINI_API_KEY')
            if key:
                return key
    # 3) try project root .streamlit/secrets.toml
    secrets_path2 = os.path.join(os.getcwd(), '.streamlit', 'secrets.toml')
    if os.path.exists(secrets_path2) and tomllib is not None:
        with open(secrets_path2, 'rb') as f:
            data = tomllib.load(f)
            key = data.get('GEMINI_API_KEY')
            if key:
                return key
    return None


def list_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            data = json.loads(body)
            return data
    except urllib.error.HTTPError as e:
        try:
            err = e.read()
            print('HTTP Error:', e.code, e.reason)
            print(err.decode('utf-8'))
        except Exception:
            print('HTTP Error:', e)
        return None
    except Exception as e:
        print('Request failed:', e)
        return None


def main():
    api_key = get_api_key()
    if not api_key:
        print('GEMINI_API_KEY not found in env or .streamlit/secrets.toml')
        sys.exit(2)
    print('Using GEMINI_API_KEY from environment or .streamlit/secrets.toml (not printed)')
    res = list_models(api_key)
    if res is None:
        print('Failed to fetch models.')
        sys.exit(1)
    # Print a concise list of model names and supportedMethods
    models = res.get('models') if isinstance(res, dict) else None
    if not models:
        print('No models field in response:')
        print(json.dumps(res, indent=2))
        sys.exit(1)
    print('\nAvailable models and supportedMethods:')
    for m in models:
        name = m.get('name')
        methods = m.get('supportedMethods') or m.get('supported_methods') or []
        print('-', name, 'methods:', methods)
    # Also print full JSON into a file for inspection
    out_path = os.path.join(os.getcwd(), 'list_models_output.json')
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'Full response written to {out_path}')

if __name__ == '__main__':
    main()
