# backend/hash_admin_users_cli.py
import json
import getpass

from config import ADMIN_USERS_PATH, ensure_storage_layout
from hash_passwords import hash_password

def main():
    ensure_storage_layout()

    # Ensure file exists
    if not ADMIN_USERS_PATH.exists():
        ADMIN_USERS_PATH.write_text('{"admins":[]}\n', encoding="utf-8")

    with open(ADMIN_USERS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    admins = data.get("admins", [])
    if not isinstance(admins, list):
        raise SystemExit("admin_users.json invalid format: expected {'admins': [...]}")

    if not admins:
        print("No admins found. Add one by typing an email below.\n")
        email = input("Admin email: ").strip().lower()
        pw = getpass.getpass(f"Enter password for {email}: ").strip()
        admins.append({"email": email, "password_hash": hash_password(pw)})
        data["admins"] = admins

        with open(ADMIN_USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ Created first admin in {ADMIN_USERS_PATH}\n")
        return

    # If admins exist, update their hashes
    for admin in admins:
        email = (admin.get("email") or "<unknown>").strip()
        pw = getpass.getpass(f"Enter password for {email}: ").strip()
        admin["password_hash"] = hash_password(pw)

        # Optional: wipe plaintext fields
        if "password" in admin:
            admin.pop("password", None)

    with open(ADMIN_USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Saved PBKDF2 password hashes to {ADMIN_USERS_PATH}\n")

if __name__ == "__main__":
    main()