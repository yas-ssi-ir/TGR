"""Installe Tailwind dans le projet, une fois pour toutes.

Pourquoi une copie locale plutot qu'un CDN : l'assistant doit repondre sans
reseau — c'est un argument du projet — et un script charge chez un tiers
signalerait chaque visite d'usager a ce tiers.

On prend @tailwindcss/browser (v4) : il compile les classes dans le
navigateur, sans etape de build, et accepte un bloc @theme pour y declarer
la palette de la TGR.
"""
import io
import os
import sys

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CIBLE = os.path.join(os.getcwd(), "static", "vendor")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
SOURCE = "https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4/dist/index.global.js"
NOM = "tailwind.js"

os.makedirs(CIBLE, exist_ok=True)
chemin = os.path.join(CIBLE, NOM)

print("Telechargement de Tailwind v4 (compilateur navigateur)…")
r = requests.get(SOURCE, headers={"User-Agent": UA}, timeout=60)
r.raise_for_status()
if len(r.content) < 100_000:
    print("ARRET — fichier suspect (%d octets), rien n'a ete ecrit." % len(r.content))
    sys.exit(1)

with open(chemin, "wb") as f:
    f.write(r.content)

print("  static/vendor/%s  %.0f Ko" % (NOM, len(r.content) / 1024))
print("\nInstalle. Les pages le chargent depuis /static/vendor/tailwind.js :")
print("plus aucune requete vers l'exterieur.")
