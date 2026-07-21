import tempfile
from pathlib import Path
import pytest

from cadre import (
    Action,
    CadreConfig,
    CadreEngine,
    DecisionStatus,
    Document,
    EvidenceState,
    RuntimeBudget,
    TrustedContext,
    TrustLevel,
    ContentOrigin,
    MessageSegment,
    EvidenceGraph,
    EvidenceEdge,
    EvidenceRelation,
    ClaimNode,
    ClaimEvidenceEdge,
    ClaimEvidenceRelation,
    ClaimGraph,
    Lineage,
    LineageRecord,
    grouped_split,
    assert_no_overlap,
    ModelArtifactManifest,
    save_artifact,
    load_artifact,
    OfflineTrainer,
)
from cadre.adapters import SimpleLLM, InMemoryRetriever


def test_trust_boundary_segmentation():
    ctx = TrustedContext(
        system="System prompt instruction.",
        developer="Dev prompt override.",
        user="User query here.",
    )
    docs = [Document(id="doc1", text="evidence doc text", source="trusted_db")]
    
    from cadre.serialization import segment_context
    segments = segment_context(ctx, docs)
    
    assert len(segments) == 4
    assert segments[0].trust == TrustLevel.TRUSTED
    assert segments[0].origin == ContentOrigin.SYSTEM
    assert segments[2].trust == TrustLevel.UNTRUSTED
    assert segments[2].origin == ContentOrigin.USER
    assert segments[3].origin == ContentOrigin.RETRIEVAL


def test_lineage_splitting_and_overlap():
    records = []
    for i in range(10):
        # 2 distinct lineage groups
        lineage = Lineage(source_id="src1", template_id=f"temp_{i % 2}")
        records.append(LineageRecord(lineage=lineage, features={"f1": 1.0}, label=1))
        
    splits = grouped_split(records, ratios=(0.5, 0.5), names=("D_fit", "D_prob"))
    assert len(splits["D_fit"]) == 5
    assert len(splits["D_prob"]) == 5
    
    # Assert no lineage overlap (should pass)
    assert_no_overlap(splits["D_fit"], splits["D_prob"])
    
    # Construct overlapping splits to test assertion
    with pytest.raises(Exception):
        assert_no_overlap(splits["D_fit"], splits["D_fit"])


def test_artifact_serialization():
    manifest = ModelArtifactManifest(
        version="1.0.0",
        code_version="0.1.0",
        data_version="0.1.0",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_obj.pkl"
        test_data = {"key": "val"}
        
        # Save
        digest = save_artifact(test_data, tmp_path, manifest=manifest)
        assert len(digest) == 64
        
        # Load and verify hash
        loaded = load_artifact(tmp_path, expected_hash=digest)
        assert loaded == test_data
        
        # Load with invalid hash raises error
        with pytest.raises(Exception):
            load_artifact(tmp_path, expected_hash="incorrect_hash")


def test_offline_training_pipeline():
    # Make D_fit, D_prob, D_threshold, D_policy, D_test
    records = []
    # 5 unique lineages
    for i in range(10):
        lineage = Lineage(source_id="src1", template_id=f"temp_{i % 5}")
        records.append(LineageRecord(lineage=lineage, features={"uncertainty": 0.1, "rank_instability": 0.2}, label=i % 2))
        
    config = CadreConfig.safe_default()
    trainer = OfflineTrainer(
        config=config,
        feature_names_by_head={
            head: config.heads[head].feature_names for head in config.heads
        }
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest = trainer.run_pipeline(records, out_dir, version="test_version")
        assert manifest.version == "test_version"
        assert (out_dir / "model_manifest.json").exists()


def test_claim_graph_and_verification():
    claims = [
        ClaimNode(id="c1", text="The refund period is 30 days."),
        ClaimNode(id="c2", text="We offer 24/7 support."),
    ]
    docs = [
        Document(id="doc1", text="We have a refund window of 30 days.", source="docs"),
    ]
    
    evidence_graph = EvidenceGraph(docs)
    
    from cadre.adapters import BasicClaimExtractor, BasicStructuredClaimVerifier
    extractor = BasicClaimExtractor()
    verifier = BasicStructuredClaimVerifier()
    
    extracted = extractor.extract("The refund period is 30 days.", metadata={})
    assert len(extracted) == 1
    
    state, features, edges = verifier.verify_claims(claims, evidence_graph, metadata={})
    
    # 30 days claim should be verified, 24/7 support should not be
    assert state == EvidenceState.INSUFFICIENT
    assert len(edges) > 0


def test_budget_exhaustion_routing():
    # Setup low budget
    budget = RuntimeBudget(
        retrieval=0,
        rewrite=0,
        rerank=0,
        graph_expansion=0,
        generation=0,
        verification=0,
        clarification=0,
        tokens=1000,
    )
    
    engine = CadreEngine(
        config=CadreConfig.safe_default(),
        llm=SimpleLLM(lambda _: "Response"),
        retriever=InMemoryRetriever([]),
    )
    
    result = engine.run(
        TrustedContext(system="Use evidence.", user="Hello"),
        budget=budget,
    )
    
    # Because budget was 0, it should abstain
    assert result.status == DecisionStatus.ABSTAINED
    assert "Maximum bounded execution depth reached" in result.reason
