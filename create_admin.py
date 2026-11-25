#!/usr/bin/env python
"""Crée un compte administrateur ou réinitialise son mot de passe.

Usage :
    python create_admin.py vous@example.com

Le mot de passe est demandé de façon masquée (deux fois). Si le compte existe
déjà, il est promu administrateur, activé, et son mot de passe est remplacé.
En environnement non interactif, la variable ADMIN_PASSWORD est utilisée.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from auth import hash_password
from database import SessionLocal, User, init_db

MIN_PASSWORD_LENGTH = 8


def _read_password() -> str:
    from_env = os.environ.get("ADMIN_PASSWORD", "")
    if from_env:
        return from_env
    if not sys.stdin.isatty():
        sys.exit("Aucun terminal interactif : définissez ADMIN_PASSWORD.")
    password = getpass.getpass("Nouveau mot de passe : ")
    if password != getpass.getpass("Confirmer le mot de passe : "):
        sys.exit("Les mots de passe ne correspondent pas.")
    return password


def main() -> None:
    parser = argparse.ArgumentParser(description="Crée ou réinitialise un compte administrateur.")
    parser.add_argument("email", help="Email du compte administrateur")
    args = parser.parse_args()

    email = args.email.strip().lower()
    if "@" not in email:
        sys.exit("Adresse email invalide.")

    password = _read_password()
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères.")

    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        created = user is None
        if created:
            user = User(email=email, full_name="Administrateur")
            db.add(user)
        user.password_hash = hash_password(password)
        user.role = "admin"
        user.status = "active"
        db.commit()
    finally:
        db.close()

    action = "créé" if created else "mis à jour (rôle admin, compte actif, nouveau mot de passe)"
    print(f"Compte administrateur {email} {action}.")


if __name__ == "__main__":
    main()
