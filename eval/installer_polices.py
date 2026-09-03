"""Installe les polices du projet dans static/fonts/.

L'assistant doit fonctionner sans reseau : une police chargee depuis Google
disparaitrait hors ligne, et signalerait chaque visite a un tiers. On les
telecharge donc UNE fois, et on les sert depuis le projet.

Sous-ensembles retenus : latin, latin-ext (francais) et arabic. Le cyrillique,
le grec et le vietnamien sont ecartes : ils ne servent a rien ici et pesent.
"""
import io
import os
import re
import sys

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RACINE = os.path.join(os.getcwd(), "static", "fonts")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
CSS = ("https://fonts.googleapis.com/css2"
       "?family=Spectral:wght@400;700"
       "&family=IBM+Plex+Sans:wght@400;600"
       "&family=IBM+Plex+Mono:wght@400;500"
       "&family=IBM+Plex+Sans+Arabic:wght@400;600"
       "&display=swap")
GARDES = {"latin", "latin-ext", "arabic"}

os.makedirs(RACINE, exist_ok=True)
css = requests.get(CSS, headers={"User-Agent": UA}, timeout=30).text

# chaque bloc est precede d'un commentaire nommant le sous-ensemble
blocs = re.findall(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]+\})", css)
sortie, vus = [], set()
for sousensemble, bloc in blocs:
    if sousensemble not in GARDES:
        continue
    famille = re.search(r"font-family:\s*'([^']+)'", bloc).group(1)
    graisse = re.search(r"font-weight:\s*(\d+)", bloc).group(1)
    url = re.search(r"url\((https://[^)]+\.woff2)\)", bloc).group(1)
    nom = "%s-%s-%s.woff2" % (famille.replace(" ", ""), graisse, sousensemble)
    chemin = os.path.join(RACINE, nom)
    if nom not in vus:
        with open(chemin, "wb") as f:
            f.write(requests.get(url, headers={"User-Agent": UA}, timeout=30).content)
        vus.add(nom)
        print("  %-42s %6.1f Ko" % (nom, os.path.getsize(chemin) / 1024))
    sortie.append(bloc
                  .replace(url, "/static/fonts/" + nom)
                  .replace("@font-face {", "@font-face{")
                  .strip())

entete = ("/* Polices du projet, servies localement.\n"
          "   Regenerable : python -X utf8 eval/installer_polices.py\n"
          "   Sous-ensembles : latin, latin-ext, arabic. */\n\n")
with open(os.path.join(RACINE, "polices.css"), "w", encoding="utf-8") as f:
    f.write(entete + "\n\n".join(sortie) + "\n")

total = sum(os.path.getsize(os.path.join(RACINE, n)) for n in vus)
print("\n%d fichiers, %.0f Ko au total" % (len(vus), total / 1024))
print("Feuille ecrite : static/fonts/polices.css")
