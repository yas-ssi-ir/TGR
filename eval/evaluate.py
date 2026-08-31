"""
Phase 6 — Évaluation comparative : RAG Classique vs Agentic RAG.

Passe le golden dataset sur les deux pipelines et mesure :
  - justesse (jugée par le LLM : la réponse contient-elle la solution attendue ?)
  - refus correct des questions hors périmètre
  - taux de fallback / hallucination apparente
  - latence

Produit : eval/RESULTATS.md (tableau comparatif) + eval/resultats_detail.json

Usage : venv\\Scripts\\python.exe eval\\evaluate.py [--only agentic|classic]
"""
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from agent_rag import AgenticRAG
from rag_classic import ClassicRAG

GOLDEN_PATH = os.path.join(BASE_DIR, "eval", "golden_dataset.json")
RESULTS_MD = os.path.join(BASE_DIR, "eval", "RESULTATS.md")
RESULTS_JSON = os.path.join(BASE_DIR, "eval", "resultats_detail.json")

JUDGE_SYSTEM = """Tu es un juge d'évaluation. Compare la RÉPONSE DONNÉE à la RÉPONSE ATTENDUE.
La réponse donnée est CORRECTE si elle contient l'information essentielle de la réponse attendue (même formulée différemment).
Réponds UNIQUEMENT par CORRECT ou INCORRECT."""

FALLBACK_MARKERS = ["ne trouve pas cette information", "hors périmètre", "je ne peux répondre",
                    "contacter le support", "problème est connu"]


def is_refusal(reponse: str) -> bool:
    low = reponse.lower()
    return any(m in low for m in ["périmètre", "ne peux répondre qu'aux questions",
                                   "ne trouve pas cette information"])


def judge(llm, reponse_donnee: str, reponse_attendue: str) -> bool:
    verdict = llm.generate(
        JUDGE_SYSTEM,
        f"RÉPONSE ATTENDUE : {reponse_attendue}\n\nRÉPONSE DONNÉE : {reponse_donnee}",
        temperature=0.0, max_tokens=8,
    ).upper()
    return "INCORRECT" not in verdict and "CORRECT" in verdict


def evaluate_pipeline(pipeline, name: str, golden: list, llm) -> dict:
    print(f"\n{'='*60}\n  Évaluation : {name}\n{'='*60}")
    rows = []
    for item in golden:
        q = item["question"]
        print(f"\n[{item['id']}] {q[:70]}")
        start = time.time()
        result = pipeline.answer(q)
        latence = round(time.time() - start, 1)
        reponse = result["reponse"]
        statut = result.get("statut", "SUCCESS")

        if item["doit_repondre"]:
            if item.get("attendu_statut") == "KNOWN_BUG":
                # attendu : orientation support, pas d'invention
                ok = statut == "KNOWN_BUG" or "support" in reponse.lower()
            else:
                ok = judge(llm, reponse, item["reponse_attendue"])
            verdict = "CORRECT" if ok else "INCORRECT"
        else:
            # question hors périmètre : le succès = refus
            ok = statut in ("OUT_OF_SCOPE", "INSUFFICIENT_KNOWLEDGE") or is_refusal(reponse)
            verdict = "REFUS_OK" if ok else "HALLUCINATION"

        print(f"    → {verdict} (statut={statut}, {latence}s)")
        rows.append({
            "id": item["id"], "question": q, "attendu": item["reponse_attendue"],
            "reponse": reponse, "statut": statut, "verdict": verdict,
            "ok": ok, "latence_s": latence,
        })
    return {"pipeline": name, "rows": rows}


def summarize(res: dict) -> dict:
    rows = res["rows"]
    in_scope = [r for r in rows if not r["id"].startswith("G1") or r["id"] in
                [f"G{i}" for i in range(1, 12)]]
    total = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    halluc = sum(1 for r in rows if r["verdict"] == "HALLUCINATION")
    lat = [r["latence_s"] for r in rows]
    return {
        "pipeline": res["pipeline"],
        "score": f"{ok}/{total}",
        "pct": round(100 * ok / total),
        "hallucinations": halluc,
        "latence_moy": round(sum(lat) / len(lat), 1),
        "latence_max": max(lat),
    }


def write_report(summaries: list[dict], details: list[dict]):
    lines = [
        "# Résultats d'évaluation — RAG Classique vs Agentic RAG",
        "",
        f"Golden dataset : {os.path.basename(GOLDEN_PATH)} — "
        f"{len(details[0]['rows'])} questions (dont 3 hors périmètre).",
        "",
        "| Pipeline | Score | % | Hallucinations | Latence moy. | Latence max |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(f"| {s['pipeline']} | {s['score']} | {s['pct']}% | "
                     f"{s['hallucinations']} | {s['latence_moy']}s | {s['latence_max']}s |")
    lines += ["", "## Détail par question", ""]
    for det in details:
        lines.append(f"### {det['pipeline']}")
        lines.append("")
        lines.append("| ID | Verdict | Statut | Latence |")
        lines.append("|---|---|---|---|")
        for r in det["rows"]:
            icon = "✅" if r["ok"] else "❌"
            lines.append(f"| {r['id']} | {icon} {r['verdict']} | {r['statut']} | {r['latence_s']}s |")
        lines.append("")
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nRapport écrit : {RESULTS_MD}")


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    print("Initialisation des pipelines (chargement E5)...")
    agent = AgenticRAG()
    classic = ClassicRAG(retriever=agent.retriever, llm=agent.llm)

    if not agent.llm.is_available():
        print("⚠ Ollama indisponible — l'évaluation nécessite le LLM.")
        sys.exit(1)

    details, summaries = [], []
    if only != "agentic":
        res_c = evaluate_pipeline(classic, "RAG Classique", golden, agent.llm)
        details.append(res_c)
        summaries.append(summarize(res_c))
    if only != "classic":
        res_a = evaluate_pipeline(agent, "Agentic RAG", golden, agent.llm)
        details.append(res_a)
        summaries.append(summarize(res_a))

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    write_report(summaries, details)

    print("\n" + "=" * 60)
    for s in summaries:
        print(f"  {s['pipeline']:<16} : {s['score']} ({s['pct']}%) "
              f"— {s['hallucinations']} hallucination(s) — {s['latence_moy']}s/question")
    print("=" * 60)


if __name__ == "__main__":
    main()
