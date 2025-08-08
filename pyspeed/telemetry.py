# In pyspeed_project/pyspeed/telemetry.py

import os
import json
import threading
import hashlib
import platform
import sys
from typing import Dict, Any

# Use a real endpoint in a production scenario
TELEMETRY_ENDPOINT = "https://your-telemetry-endpoint.com/collect"
CONFIG_DIR = os.path.expanduser("~/.pyspeed")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def _get_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}

def _save_config(config: Dict[str, Any]):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not write PySpeed config to {CONFIG_FILE}: {e}")

def ask_for_consent():
    config = _get_config()
    if "telemetry_consent" in config:
        return config["telemetry_consent"]

    print("-" * 60)
    print("Welcome to PySpeed!")
    print("\nTo help improve this tool, you can opt-in to sending anonymous")
    print("usage data. This includes:")
    print("  - Anonymized source code hash (not the code itself)")
    print("  - Optimization suggestions and outcomes")
    print("  - Performance improvements (timing data)")
    print("  - Python version and OS type")
    print("\nThis data is fully anonymous and helps us build a better optimizer.")
    
    answer = input("Do you agree to send anonymous usage data? (yes/no): ").lower().strip()
    print("-" * 60)
    
    consent = answer in ["y", "yes"]
    config["telemetry_consent"] = consent
    _save_config(config)
    
    if consent:
        print("Thank you! Telemetry is enabled.")
    else:
        print("Telemetry is disabled.")
    
    return consent

def user_consents_to_telemetry() -> bool:
    config = _get_config()
    return config.get("telemetry_consent", False)

def build_payload(source_code: str, analysis_result, timing_result: Dict) -> Dict:
    source_hash = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
    payload = {
        "source_hash": source_hash,
        "suggestions": [s.__dict__ for s in analysis_result.suggestions],
        "decorated_funcs": analysis_result.decorated_funcs,
        "timing": timing_result,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    return payload

def _upload_data(payload: Dict):
    """The actual upload function that runs in a thread."""
    try:
        import requests
        headers = {'Content-Type': 'application/json'}
        response = requests.post(TELEMETRY_ENDPOINT, data=json.dumps(payload), headers=headers, timeout=10)
        # In a real app, you might log the status code for debugging
        print(f"Telemetry response: {response.status_code}")
    except Exception as e:
        # Silently fail if telemetry upload fails. Don't interrupt the user.
        print(f"Telemetry upload failed: {e}")

def start_telemetry_upload_thread(payload: Dict):
    if not user_consents_to_telemetry():
        return
    thread = threading.Thread(target=_upload_data, args=(payload,), daemon=True)
    thread.start()
