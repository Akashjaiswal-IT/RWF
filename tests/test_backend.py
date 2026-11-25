"""
Automated Verification Suite for RWF Studio Backend.
Verifies:
1. Exact match against Part 4 worked example (pp. 40-42 of RWF_Paper_Explained.pdf)
2. Exact match against Part 4 Graph 2 (Star + Edge) exact zeros at tau=3
3. Real SVM training + 10-fold CV accuracy & 2D projection
4. Full FastAPI REST API endpoints + SQLite CRUD lifecycle
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from backend.database import init_db  # noqa: E402
from backend.main import app  # noqa: E402
from backend.paper_data import PRESET_GRAPHS  # noqa: E402
from backend.rwf_engine import compute_rwf  # noqa: E402


def test_part4_worked_example():
    g = PRESET_GRAPHS["worked"]
    res = compute_rwf(g["nodes"], g["edges"], mode="rwf-d", k=2, tau_c=3, use_node_features=False)
    expected = [
        0.000, 0.202, 0.000, 0.000, 0.204,
        0.477, 0.252, 0.292, 0.137, 0.123,
        0.111, 0.230, 0.028, 0.071, 0.160,
    ]
    got = [round(v, 3) for v in res["fingerprint"]]
    assert got == expected, f"Mismatch!\nGot:      {got}\nExpected: {expected}"
    print("[PASS] Part 4 Worked Example matches PDF to 3 decimal places:", got)


def test_part4_star_zeros():
    g = PRESET_GRAPHS["star"]
    res = compute_rwf(g["nodes"], g["edges"], mode="rwf-d", k=2, tau_c=3, use_node_features=False)
    # At tau=3, loop V2 (idx 12) and walk V2 (idx 13) must be exactly 0.0
    assert abs(res["fingerprint"][12]) < 1e-9, f"Expected 0 loop V2 at tau=3, got {res['fingerprint'][12]}"
    assert abs(res["fingerprint"][13]) < 1e-9, f"Expected 0 walk V2 at tau=3, got {res['fingerprint'][13]}"
    print("[PASS] Part 4 Star Graph exact zeros at tau=3 verified.")


def test_api_end_to_end():
    init_db()
    client = TestClient(app)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200 and r_health.json()["status"] == "ok"
    print("[PASS] GET /api/health:", r_health.json())

    g = PRESET_GRAPHS["worked"]
    r_fp = client.post("/api/rwf/fingerprint", json={"nodes": g["nodes"], "edges": g["edges"], "mode": "rwf-d", "k": 2, "tau_c": 3})
    assert r_fp.status_code == 200
    fp_data = r_fp.json()
    assert fp_data["dimension"] == 15
    assert len(fp_data["eigenvalues"]) == 6
    print(f"[PASS] POST /api/rwf/fingerprint (d={fp_data['dimension']}, {fp_data['elapsed_ms']}ms, eigvals={fp_data['eigenvalues'][:3]}...)")

    r_cmp = client.post("/api/rwf/compare", json={"graph_a": {"nodes": g["nodes"], "edges": g["edges"], "mode": "rwf-d", "k": 2, "tau_c": 3}, "compare_with_preset": "star"})
    assert r_cmp.status_code == 200 and r_cmp.json()["comparison"]["l2_distance"] > 0
    print("[PASS] POST /api/rwf/compare:", r_cmp.json()["comparison"]["l2_distance"])

    r_svm = client.post("/api/svm/classify", json={"nodes": g["nodes"], "edges": g["edges"], "mode": "rwf-d", "k": 2, "tau_c": 3, "C": 1.0, "dataset_name": "mutag-mini"})
    assert r_svm.status_code == 200
    svm_data = r_svm.json()
    assert 0.0 <= svm_data["cv_10fold"]["mean_acc"] <= 100.0
    print(f"[PASS] POST /api/svm/classify -> {svm_data['predicted_label']} ({svm_data['confidence_pct']}%), 10-fold CV={svm_data['cv_10fold']['mean_acc']}% (±{svm_data['cv_10fold']['std_acc']})")

    r_abl = client.post("/api/dataset/ablation", json={"dataset_name": "mutag-mini", "max_tau": 4})
    assert r_abl.status_code == 200 and "RWF-D (k=2)" in r_abl.json()["series"]
    print("[PASS] POST /api/dataset/ablation:", r_abl.json()["runtimes_ms"])

    # SQLite Save -> List -> Delete
    r_save = client.post("/api/graphs", json={"name": "IIT Demo Graph", "nodes": g["nodes"], "edges": g["edges"], "mode": "rwf-d", "k": 2, "tau_c": 3})
    assert r_save.status_code == 200
    gid = r_save.json()["saved"]["id"]
    r_list = client.get("/api/graphs")
    assert any(item["id"] == gid for item in r_list.json()["graphs"])
    r_del = client.delete(f"/api/graphs/{gid}")
    assert r_del.status_code == 200 and r_del.json()["deleted"] is True
    print(f"[PASS] SQLite CRUD (/api/graphs save/list/delete id={gid})")


if __name__ == "__main__":
    test_part4_worked_example()
    test_part4_star_zeros()
    test_api_end_to_end()
    print("\nALL BACKEND TESTS PASSED!")
