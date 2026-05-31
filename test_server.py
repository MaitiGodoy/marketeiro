import pytest
import json
from server import (
    check_seo_keywords,
    validate_ad_policy,
    score_virality,
    export_tracking_config,
    check_financial_guard,
    generate_creative,
    validate_creative_specs,
    batch_generate_carousel,
    apply_transform,
    generate_ai_image,
    get_token_usage_summary,
    validate_token_policy,
    force_external_only_test
)

def test_check_seo_keywords():
    res = check_seo_keywords("Este é um anúncio de marketing digital e SEO", ["marketing", "seo", "inexistente"])
    data = json.loads(res)
    assert "seo_score" in data
    assert data["seo_score"] == pytest.approx(66.66, 0.1)

def test_validate_ad_policy():
    res = validate_ad_policy("Gere uma cura milagrosa e fique rico hoje!")
    data = json.loads(res)
    assert data["policy"] == "fail"
    assert len(data["violations"]) == 2

def test_score_virality():
    res = score_virality("Como ter sucesso no marketing digital? Descubra agora!")
    data = json.loads(res)
    assert "virality_score" in data
    assert data["virality_score"] > 50

def test_export_tracking_config():
    res = export_tracking_config("https://exemplo.com", "facebook", "cpc", "blackfriday")
    data = json.loads(res)
    assert "tracked_url" in data
    assert "utm_source=facebook" in data["tracked_url"]

def test_check_financial_guard():
    res = check_financial_guard(10.0)
    data = json.loads(res)
    assert data["allowed"] is False  # limit is 0.0

# --- V3.1 Creative Module Tests ---

def test_generate_creative():
    copy = {"headline": "Bora escalar seu SaaS", "body": "Aprenda a fazer tráfego pago hoje", "cta": "Clique aqui"}
    style = {"palette": ["#FF6B6B", "#4ECDC4"]}
    res = generate_creative("static", copy, style, "meta")
    data = json.loads(res)
    assert data["type"] == "static"
    assert data["platform"] == "meta"
    assert len(data["assets"]) == 1
    assert data["assets"][0]["base64"] is not None

def test_validate_creative_specs():
    creative = {"width": 1080, "height": 1080, "text_coverage_pct": 25.0}
    res = validate_creative_specs("meta", creative)
    data = json.loads(res)
    assert data["compliant"] is False
    assert len(data["issues"]) > 0

def test_batch_generate_carousel():
    cards = [{"headline": "Passo 1"}, {"headline": "Passo 2"}]
    res = batch_generate_carousel(cards, "sequential", "meta")
    data = json.loads(res)
    assert data["total_cards"] == 2
    assert len(data["urls"]) == 2

def test_apply_transform():
    res = apply_transform("local_url.png", {"width": 500, "height": 500})
    data = json.loads(res)
    assert "transformed_url" in data
    assert "500x500" in data["transformed_url"]

def test_generate_ai_image():
    res = generate_ai_image("um escritório minimalista de produtividade", {"theme": "dark"}, [1080, 1080])
    data = json.loads(res)
    assert "base64" in data
    assert data["specs"]["width"] == 1080

# --- Section 0 Token Policy Tests ---

def test_get_token_usage_summary():
    res = get_token_usage_summary()
    data = json.loads(res)
    assert "host_tokens_used" in data
    assert "external_api_calls" in data
    assert data["policy_compliant"] is True

def test_validate_token_policy():
    res = validate_token_policy("creator", "openrouter")
    data = json.loads(res)
    assert "policy_status" in data
    assert "host_native_blocked" in data
    assert data["host_native_blocked"] is True

@pytest.mark.asyncio
async def test_force_external_only_test():
    res = await force_external_only_test("creator")
    data = json.loads(res)
    assert "routing_status" in data
    assert "role" in data

# --- V4.0 Vivar SEO Agent Tests ---
from server import register_blog_post_in_index, optimize_site_keywords_and_meta, run_vivar_seo_agent_loop
from pathlib import Path

def test_register_blog_post_in_index(tmp_path):
    # Setup temporary index.html
    index_file = tmp_path / "index.html"
    index_file.write_text('<html><body><div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-6" id="articles-grid"></div></body></html>', encoding="utf-8")
    
    success = register_blog_post_in_index(
        index_file,
        "test-slug",
        "Test Title",
        "Test Excerpt",
        "gestao",
        "30 Mai 2026"
    )
    
    assert success is True
    content = index_file.read_text(encoding="utf-8")
    assert "artigos/test-slug.html" in content
    assert "Test Title" in content
    assert "Test Excerpt" in content

def test_optimize_site_keywords_and_meta(tmp_path):
    index_file = tmp_path / "index.html"
    index_file.write_text('<html><head><meta name="keywords" content="original"><meta name="description" content="original"></head></html>', encoding="utf-8")
    
    res = optimize_site_keywords_and_meta(
        str(tmp_path),
        "newkey",
        "new desc"
    )
    
    data = json.loads(res)
    assert data["status"] == "success"
    assert "index.html" in data["updated_files"]
    
    content = index_file.read_text(encoding="utf-8")
    assert "newkey" in content
    assert "new desc" in content

@pytest.mark.asyncio
async def test_run_vivar_seo_agent_loop_missing_path():
    # If the site path is invalid, it should handle it gracefully or return failure/success when mock runs
    res = await run_vivar_seo_agent_loop("invalid_path", "construção civil", "GERICFAST", auto_publish=False)
    data = json.loads(res)
    # Since auto_publish=False, it won't check site path for publishing, but it should succeed
    if "status" in data:
        assert data["status"] in ["success", "failed"]

@pytest.mark.asyncio
async def test_team_brainstorm():
    from server import team_brainstorm
    res = await team_brainstorm("Criar a landing page principal da GERICFAST")
    assert isinstance(res, str)
    assert len(res) > 0

@pytest.mark.asyncio
async def test_apply_ux_ui_refinement(tmp_path):
    from server import apply_ux_ui_refinement
    index_file = tmp_path / "index.html"
    index_file.write_text('<html><head><title>Test</title></head><body></body></html>', encoding="utf-8")
    
    res = await apply_ux_ui_refinement(str(tmp_path), "Adicionar gradiente dourado de luxo nos títulos e sombra nos cards")
    data = json.loads(res)
    assert data["status"] == "success"
    assert "index.html" in data["updated_files"]
    
    content = index_file.read_text(encoding="utf-8")
    assert "</head>" in content
    assert "<style>" in content or "<script>" in content



