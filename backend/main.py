"""
Production-Ready FastAPI Backend for RWF Monograph (IEEE ICDM 2024).
Compatible with Pydantic v1 & v2, FastAPI 0.92+, Render, Railway, Docker, and local runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .database import DB_PATH, delete_graph, init_db, list_experiments, list_graphs, log_experiment, save_graph
from .paper_data import PRESET_GRAPHS, RQ1_BENCHMARKS, RQ3_RUNTIMES
from .rwf_engine import compare_fingerprints, compute_rwf
from .svm_classifier import HAS_SKLEARN, classify_user_graph, run_live_ablation

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _to_dict(model: BaseModel) -> Dict[str, Any]:
    """Cross-compatible Pydantic v1 (.dict) and v2 (.model_dump) serializer."""
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


app = FastAPI(
    title="RWF Monograph API",
    description="Real NumPy/SciPy Random Walk Fingerprints (IEEE ICDM 2024) + Soft-Margin SVM & SQLite Backend",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure SQLite schema exists immediately on import & startup
init_db()


class NodeItem(BaseModel):
    id: str
    x: Optional[float] = 300.0
    y: Optional[float] = 180.0
    feat: Optional[str] = "C"


class FingerprintRequest(BaseModel):
    nodes: List[NodeItem]
    edges: List[List[str]]
    mode: str = Field(default="rwf-d", description="rwf-d | rwf-cp | rwf-w | rwf-r")
    k: int = Field(default=2, ge=1, le=5)
    tau_c: int = Field(default=3, ge=1, le=10)
    use_node_features: bool = False


class CompareRequest(BaseModel):
    graph_a: FingerprintRequest
    graph_b: Optional[FingerprintRequest] = None
    compare_with_preset: Optional[str] = "star"


class SvmClassifyRequest(FingerprintRequest):
    C: float = Field(default=1.0, gt=0.0, le=10000.0)
    dataset_name: str = Field(default="mutag-mini", description="mutag-mini | reddit-mini")


class SaveGraphRequest(FingerprintRequest):
    name: str = Field(..., min_length=1, max_length=80)


class AblationRequest(BaseModel):
    dataset_name: str = Field(default="mutag-mini")
    max_tau: int = Field(default=6, ge=2, le=8)


@app.get("/", include_in_schema=False)
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"message": "RWF Monograph API is running. Place frontend at static/index.html."})
    return FileResponse(str(index_path))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/healthz", include_in_schema=False)
def healthz() -> Dict[str, str]:
    """Standard cloud health-check endpoint (Render / Railway / Docker)."""
    return {"status": "ok"}


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "RWF Monograph Backend",
        "svm_engine": "scikit-learn SVC" if HAS_SKLEARN else "SciPy L-BFGS-B Soft-Margin SVM",
        "db_path": str(DB_PATH.name),
        "saved_graphs_count": len(list_graphs()),
    }


@app.get("/api/paper/presets")
def get_presets() -> Dict[str, Any]:
    return {"presets": PRESET_GRAPHS}


@app.get("/api/paper/benchmarks")
def get_paper_benchmarks() -> Dict[str, Any]:
    return {"rq1": RQ1_BENCHMARKS, "rq3": RQ3_RUNTIMES}


@app.post("/api/rwf/fingerprint")
def api_fingerprint(req: FingerprintRequest) -> Dict[str, Any]:
    nodes = [_to_dict(n) for n in req.nodes]
    return compute_rwf(
        nodes=nodes,
        edges=req.edges,
        mode=req.mode,
        k=req.k,
        tau_c=req.tau_c,
        use_node_features=req.use_node_features,
        include_matrices=True,
    )


@app.post("/api/rwf/compare")
def api_compare(req: CompareRequest) -> Dict[str, Any]:
    res_a = compute_rwf(
        nodes=[_to_dict(n) for n in req.graph_a.nodes],
        edges=req.graph_a.edges,
        mode=req.graph_a.mode,
        k=req.graph_a.k,
        tau_c=req.graph_a.tau_c,
        use_node_features=req.graph_a.use_node_features,
        include_matrices=False,
    )
    if req.graph_b is not None:
        res_b = compute_rwf(
            nodes=[_to_dict(n) for n in req.graph_b.nodes],
            edges=req.graph_b.edges,
            mode=req.graph_b.mode,
            k=req.graph_b.k,
            tau_c=req.graph_b.tau_c,
            use_node_features=req.graph_b.use_node_features,
            include_matrices=False,
        )
    else:
        preset_key = req.compare_with_preset if req.compare_with_preset in PRESET_GRAPHS else "star"
        preset = PRESET_GRAPHS[preset_key]
        res_b = compute_rwf(
            nodes=preset["nodes"],
            edges=preset["edges"],
            mode=req.graph_a.mode,
            k=req.graph_a.k,
            tau_c=req.graph_a.tau_c,
            use_node_features=req.graph_a.use_node_features,
            include_matrices=False,
        )

    return {
        "graph_a": res_a,
        "graph_b": res_b,
        "comparison": compare_fingerprints(res_a["fingerprint"], res_b["fingerprint"]),
    }


@app.post("/api/svm/classify")
def api_svm_classify(req: SvmClassifyRequest) -> Dict[str, Any]:
    nodes = [_to_dict(n) for n in req.nodes]
    res = classify_user_graph(
        nodes=nodes,
        edges=req.edges,
        mode=req.mode,
        k=req.k,
        tau_c=req.tau_c,
        use_node_features=req.use_node_features,
        C=req.C,
        dataset_name=req.dataset_name,
    )
    log_experiment(
        kind="svm_classify",
        summary=f"{req.mode.upper()} (k={req.k}, tau_c={req.tau_c}, C={req.C}) -> {res['predicted_label']} ({res['confidence_pct']}%)",
        payload={"dataset": req.dataset_name, "cv_mean": res["cv_10fold"]["mean_acc"], "pred": res["predicted_class"]},
    )
    return res


@app.post("/api/dataset/ablation")
def api_ablation(req: AblationRequest) -> Dict[str, Any]:
    res = run_live_ablation(dataset_name=req.dataset_name, max_tau=req.max_tau)
    log_experiment(
        kind="live_ablation",
        summary=f"Live Ablation on {req.dataset_name} (tau=1..{req.max_tau})",
        payload=res["runtimes_ms"],
    )
    return res


@app.get("/api/graphs")
def api_list_graphs() -> Dict[str, Any]:
    return {"graphs": list_graphs()}


@app.post("/api/graphs")
def api_save_graph(req: SaveGraphRequest) -> Dict[str, Any]:
    nodes = [_to_dict(n) for n in req.nodes]
    rwf = compute_rwf(
        nodes=nodes,
        edges=req.edges,
        mode=req.mode,
        k=req.k,
        tau_c=req.tau_c,
        use_node_features=req.use_node_features,
        include_matrices=False,
    )
    saved = save_graph(
        name=req.name.strip(),
        nodes=nodes,
        edges=req.edges,
        mode=req.mode,
        k=req.k,
        tau_c=req.tau_c,
        fingerprint=rwf["fingerprint"],
    )
    return {"saved": saved, "dimension": rwf["dimension"]}


@app.delete("/api/graphs/{graph_id}")
def api_delete_graph(graph_id: int) -> Dict[str, Any]:
    if not delete_graph(graph_id):
        raise HTTPException(status_code=404, detail="Graph not found")
    return {"deleted": True, "id": graph_id}


@app.get("/api/experiments")
def api_list_experiments() -> Dict[str, Any]:
    return {"experiments": list_experiments()}
