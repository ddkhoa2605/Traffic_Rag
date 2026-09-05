from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.graph.config import GraphSettings, _dotenv
from src.retrieval_runtime import PostgresB6Application, application_health

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=10)
    polish: bool = False


def demo_gemini_model(root: Path) -> str:
    values = {**_dotenv(root / ".env"), **_dotenv(root / ".env.lightrag")}
    return os.environ.get("DEMO_GEMINI_MODEL") or values.get("DEMO_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def hierarchy_label(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        return value
    labels = (
        ("part", "Phần"), ("chapter", "Chương"), ("section", "Mục"),
        ("subsection", "Tiểu mục"), ("article", "Điều"), ("clause", "Khoản"),
        ("point", "Điểm"), ("subpoint", "Ý"),
    )
    return " · ".join(f"{label} {value[key]}" for key, label in labels if value.get(key)) or "Văn bản liên quan"


def build_polish_prompt(query: str, results: list[dict[str, Any]]) -> str:
    evidence = "\n\n".join(
        "\n".join((
            f"[Kết quả {item['rank']}]",
            f"Nguồn: {item['document_id']} — {hierarchy_label(item['hierarchy'])}",
            f"Citation IDs: {', '.join(item['citation_node_ids'])}",
            f"Evidence:\n{item['evidence_text']}",
        ))
        for item in results[:5]
    )
    return f"""Bạn là lớp trình bày cho demo tra cứu luật giao thông Việt Nam.
Chỉ dùng Evidence bên dưới; không suy diễn, không bổ sung kiến thức ngoài evidence,
không đưa tư vấn pháp lý. Nếu evidence không đủ để trả lời, nói rõ điều đó.
Trả lời bằng tiếng Việt, ngắn gọn, dễ đọc. Cuối mỗi ý có căn cứ phải giữ nguyên ít nhất
một citation node ID trong ngoặc vuông, ví dụ [LAW_36_2024:ARTICLE:2].

Câu hỏi: {query}

Evidence:
{evidence}"""


def polish_with_gemini(query: str, results: list[dict[str, Any]], api_key: str, model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError('Thiếu Google GenAI SDK. Chạy: python -m pip install -e ".[demo]"') from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_polish_prompt(query, results),
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=500),
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Gemini không trả về nội dung")
    allowed = {citation for item in results for citation in item["citation_node_ids"]}
    cited = set(re.findall(r"\[([^\]]+)\]", answer))
    unknown = cited - allowed
    if unknown:
        raise RuntimeError(f"Gemini trả citation ngoài evidence: {', '.join(sorted(unknown))}")
    if not cited and allowed:
        answer += "\n\nCăn cứ: " + " ".join(f"[{citation}]" for citation in sorted(allowed))
    return answer


def page() -> str:
    return """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tra cứu Luật Giao thông</title><style>
:root{--ink:#142033;--muted:#60708a;--line:#dce4ef;--surface:#fff;--canvas:#f5f7fb;--brand:#1463d8;--soft:#eaf2ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#f8fbff 0,#f5f7fb 280px);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1060px;margin:auto;padding:50px 20px 72px}h1{font-size:32px;line-height:1.2;margin:0}h2{font-size:18px;margin:0 0 8px}.meta{color:var(--muted);margin:0}.panel,.result{background:var(--surface);border:1px solid var(--line);border-radius:16px;box-shadow:0 5px 18px #1622350b}.panel{margin-top:26px;padding:22px}.panel h2{font-size:17px}textarea{width:100%;resize:vertical;min-height:94px;padding:13px;border:1px solid #b8c6d9;border-radius:11px;font:inherit;color:inherit}textarea:focus{outline:3px solid #cfe1ff;border-color:var(--brand)}.row{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-top:14px}label{color:#34445d}select{padding:8px;border:1px solid #b8c6d9;border-radius:8px;background:#fff}.gemini-toggle{display:flex;align-items:center;gap:7px;font-weight:600}.gemini-toggle input{width:16px;height:16px;accent-color:var(--brand)}button{background:var(--brand);color:#fff;border:0;border-radius:9px;padding:10px 17px;font-weight:700;cursor:pointer}button:disabled{opacity:.65;cursor:wait}.notice{margin-top:14px;padding:10px 12px;background:var(--soft);border-radius:9px;color:#24509b}.error{background:#fff0f0;color:#a52020}.hidden{display:none}.summary{margin:28px 0 13px;font-weight:650}.result{padding:19px;margin:13px 0}.result-head{display:flex;justify-content:space-between;gap:12px;align-items:start}.badge{display:inline-block;background:#e8f0ff;color:#1553aa;border-radius:99px;padding:3px 9px;font-size:12px;font-weight:750;margin-bottom:7px}.score{color:var(--muted);font-variant-numeric:tabular-nums}.evidence{white-space:pre-wrap;background:#f8fafc;border-left:3px solid #79a9f7;padding:12px 14px;margin:14px 0 0}.citations{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:#315c9b;overflow-wrap:anywhere;margin-top:10px}.answer{white-space:pre-wrap;background:#f1f8f4;border:1px solid #b9e0c5;border-radius:11px;padding:17px;margin:15px 0}.footer{color:var(--muted);font-size:13px;margin-top:30px}@media(max-width:600px){main{padding-top:32px}.result-head{display:block}}
</style></head><body><main>
<header><h1>Tra cứu Luật Giao thông</h1></header>
<section class="panel"><h2>Đặt câu hỏi</h2><textarea id="query" placeholder="Ví dụ: Khoản 1 Điều 2 quy định nội dung gì?"></textarea><div class="row"><label>Top-k <select id="topk"><option>3</option><option selected>5</option></select></label><label class="gemini-toggle"><input id="polish" type="checkbox"> Sử dụng Gemini để generate</label><button id="submit">Tra cứu</button></div><div id="message" class="notice hidden"></div></section>
<section id="output" class="hidden"><div id="summary" class="summary"></div><div id="answer"></div><div id="results"></div></section>
</main><script>
const $=id=>document.getElementById(id);const text=v=>String(v??'');
function notice(value,error=false){const box=$('message');box.textContent=value;box.className='notice'+(error?' error':'');}
function hierarchyText(value){if(typeof value==='string')return value;if(!value||typeof value!=='object')return 'Văn bản liên quan';const labels=[['part','Phần'],['chapter','Chương'],['section','Mục'],['subsection','Tiểu mục'],['article','Điều'],['clause','Khoản'],['point','Điểm'],['subpoint','Ý']];const parts=labels.filter(([key])=>value[key]).map(([key,label])=>label+' '+value[key]);return parts.join(' · ')||'Văn bản liên quan'}
function card(item){const el=document.createElement('article');el.className='result';const head=document.createElement('div');head.className='result-head';const left=document.createElement('div');const badge=document.createElement('span');badge.className='badge';badge.textContent='Top '+text(item.rank);const h=document.createElement('h2');h.textContent=hierarchyText(item.hierarchy);const meta=document.createElement('div');meta.className='meta';meta.textContent=text(item.source_strategy)+' · '+text(item.document_id)+' · '+text(item.result_type);left.append(badge,h,meta);const score=document.createElement('div');score.className='score';score.textContent='Score '+Number(item.score).toFixed(3);head.append(left,score);const evidence=document.createElement('div');evidence.className='evidence';evidence.textContent=item.evidence_text;const cites=document.createElement('div');cites.className='citations';cites.textContent='Căn cứ: '+item.citation_node_ids.join(', ');el.append(head,evidence,cites);return el}
async function search(){const query=$('query').value.trim();if(query.length<3){notice('Hãy nhập câu hỏi ít nhất 3 ký tự.',true);return}const button=$('submit');button.disabled=true;notice('Đang tìm evidence…');$('output').classList.add('hidden');try{const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:query,top_k:Number($('topk').value),polish:$('polish').checked})});const data=await r.json();if(!r.ok)throw new Error(data.detail||'Không thể tra cứu');$('results').replaceChildren(...data.results.map(card));$('summary').textContent='Tìm thấy '+data.results.length+' kết quả tham chiếu.';const holder=$('answer');holder.replaceChildren();if(data.polished_answer){const h=document.createElement('h2');h.textContent='Câu trả lời của system: ';const a=document.createElement('div');a.className='answer';a.textContent=data.polished_answer;holder.append(h,a)}notice(data.polish_notice||'Đã tra cứu xong.');$('output').classList.remove('hidden')}catch(e){notice(e.message,true)}finally{button.disabled=false}}
$('submit').addEventListener('click',search);$('query').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')search()});
</script></body></html>"""


def create_app(root: str | Path | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError('Thiếu web dependencies. Chạy: python -m pip install -e ".[demo]"') from exc

    root_path = Path(root or Path.cwd()).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retrieval = None
        app.state.settings = GraphSettings.load(root_path)
        yield
        if app.state.retrieval is not None:
            app.state.retrieval.close()

    app = FastAPI(title="Traffic Law RAG Demo", version="0.1.0", lifespan=lifespan)

    def retrieval() -> PostgresB6Application:
        if app.state.retrieval is None:
            app.state.retrieval = PostgresB6Application(root_path)
        return app.state.retrieval

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return page()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        payload = application_health(root_path)
        payload["gemini_available"] = bool(app.state.settings.gemini_api_key)
        payload["gemini_model"] = demo_gemini_model(root_path)
        return payload

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            payload = retrieval().search(request.query, top_k=request.top_k).model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Retrieval chưa sẵn sàng: {exc}") from exc

        answer = None
        notice = "Đã truy xuất evidence; không gọi Gemini."
        if request.polish:
            api_key = app.state.settings.gemini_api_key
            if not api_key:
                raise HTTPException(status_code=400, detail="Gemini chưa được cấu hình. Thêm GEMINI_API_KEY vào .env.lightrag.")
            model = demo_gemini_model(root_path)
            try:
                answer = polish_with_gemini(request.query, payload["results"], api_key, model)
                notice = f"Gemini đã generate câu trả lời từ top {len(payload['results'])} evidence; hãy kiểm tra căn cứ bên dưới."
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Gemini không thể trả lời: {exc}") from exc
        return {**payload, "polished_answer": answer, "polish_notice": notice}

    return app

