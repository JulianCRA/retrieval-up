import sys
import subprocess
import json
from pathlib import Path

EMBEDDERS = ['bge-m3', 'e5-large-instruct', 'granite-107m', 'jina-v3', 'qwen3-0.6b']
RERANKERS = ['mmarco', 'bge', 'jina']
MODO = 'rrf'
DATASET = r'C:\Users\ccjj\Desktop\retrieval-up\eval_dataset\dataset.json'
OUTPUT_FILE = 'all_results.json'

def main():
    print("Elige la estrategia de fragmentación (1 o 2):")
    print("1) semantico")
    print("2) tamano_fijo")
    opcion = input("> ").strip()
    
    estrategia_args = []
    metadata = {}
    
    if opcion == '1':
        estrategia = 'semantico'
        umbral = input("Introduce el umbral (ej. 0.3): ").strip()
        min_tokens = input("Introduce min_tokens (ej. 100): ").strip()
        estrategia_args = ["--chunk-estrategia", "semantico", "--chunk-umbral", umbral]
        if min_tokens:
            estrategia_args.extend(["--chunk-min-tokens", min_tokens])
        metadata = {"estrategia": "semantico", "umbral": float(umbral) if umbral else None, "min_tokens": int(min_tokens) if min_tokens else None}
    else:
        estrategia = 'tamano_fijo'
        overlap = input("Introduce el solapamiento en porcentaje (ej. 10): ").strip()
        estrategia_args = ["--chunk-estrategia", "tamano_fijo", "--chunk-overlap-pct", overlap]
        metadata = {"estrategia": "tamano_fijo", "overlap_pct": int(overlap) if overlap else None}

    final_results = {
        "dataset": DATASET,
        "chunking": metadata,
        "runs": []
    }
    
    for emb in EMBEDDERS:
        extracted_baseline = False
        for rerank in RERANKERS:
            cmd = [
                sys.executable, "evaluar_ret.py",
                "--dataset", DATASET,
                "--embedder", emb,
                "--modo", MODO
            ]
            cmd.extend(estrategia_args)
            
            if rerank:
                cmd.extend(["--reranker", rerank])
            
            print(f"\n[{emb} + {rerank or 'none'}]")
            print(f"Running: {' '.join(cmd)}")
            
            # Use subprocess.run to execute the script
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            
            # stdout/stderr could be None if it completely failed without capturing
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            
            # Find the path printed by the script
            # Example: [OK] Guardado en: C:\Users\ccjj\Desktop\retrieval-up\resultados\ret_eval\20260622_160549_bge-m3-rrf-norerank.json
            path_line = [line for line in stdout.splitlines() if "[OK] Guardado en:" in line]
            
            if path_line and result.returncode == 0:
                json_path = path_line[0].split("Guardado en:")[1].strip()
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        run_data = json.load(f)
                        summary = run_data.get("summary", {})
                        
                        if not extracted_baseline and "pre_rerank" in summary:
                            base_summary = {
                                "num_queries": summary.get("num_queries"),
                                "num_ok": summary.get("num_ok"),
                                "num_failed": summary.get("num_failed")
                            }
                            # Update with the pre_rerank metrics
                            base_summary.update(summary["pre_rerank"])
                            final_results["runs"].append({
                                "embedder": emb,
                                "reranker": None,
                                "summary": base_summary
                            })
                            extracted_baseline = True
                        
                        # Remove pre_rerank so it doesn't duplicate the data inside the reranker run
                        if "pre_rerank" in summary:
                            del summary["pre_rerank"]
                        
                        final_results["runs"].append({
                            "embedder": emb,
                            "reranker": rerank,
                            "summary": summary
                        })
                        print(f"Success. Loaded summary from: {json_path}")
                except Exception as e:
                     print(f"Failed to read JSON at {json_path}: {e}")
            else:
                print(f"Error running combination. Exit code: {result.returncode}")
                if stderr:
                    print(f"STDERR:\n{stderr}")
                if stdout:
                    print(f"STDOUT:\n{stdout}")
    
    # Save the consolidated results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] All results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
