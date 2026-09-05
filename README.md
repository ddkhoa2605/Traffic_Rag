# Traffic Law Parser

Pipeline tạo canonical legal tree cho Luật 35/2024/QH15 và Luật 36/2024/QH15:

```text
registry → PDF inspect/render → native extraction → normalization
→ legal state machine → validation → validated corpus
```

## Cài đặt

```powershell
python -m pip install -e ".[test]"
```

## Build

```powershell
python -m src.cli.inspect LAW_35_2024
python -m src.cli.extract LAW_35_2024
python -m src.cli.parse LAW_35_2024
python -m src.cli.validate LAW_35_2024
python -m src.cli.build --all
```

Dùng `--no-render` để bỏ qua PNG khi chạy nhanh. Build chỉ publish vào
`data/05_validated` khi validation PASS; ERROR/FATAL đi vào quarantine. WARN cần
`--approve-warnings` sau manual review.

## Đọc corpus

```python
from src.corpus import load_validated_corpus

corpus = load_validated_corpus()
law36 = corpus.document("LAW_36_2024")
point = law36.article("11").clause("2").point("đ")

print(point.text)
print(point.source.page_start)
print(point.source.blocks)
```

## Test

```powershell
python -m pytest
```

Gold suite gồm 25 case cho mỗi luật, bao phủ Điều/Khoản/Điểm, `đ)`, node đa
trang, header và fidelity. Regression snapshot được tạo ở
`reports/<document_id>/regression_snapshot.jsonl` sau mỗi build.

## Expanded legal tree và chunking ablation

Canonical nodes vẫn là nguồn sự thật và không bị mutate. `src/legal_tree` cung cấp
resolver để duyệt parent/child/ancestor/descendant; `src/chunking` chiếu cùng corpus
thành sáu dạng passage rebuildable B0-B5. Citation luôn dùng canonical node ID.

```text
validated canonical tree
  -> B0-B5 retrieval passages
  -> BM25 hoặc BGE-M3 dense retrieval
  -> B5 evidence expansion (khi áp dụng)
  -> exact/structural/evidence metrics
```

Các lệnh chính:

```powershell
python -m pip install -e ".[test,eval]"
python -m src.cli.build_passages --all
python -m src.cli.benchmark_dataset --validate
python -m src.cli.eval_chunks --all --retriever bm25 --split dev
python -m src.cli.eval_chunks --all --retriever dense --split dev
python -m src.cli.compare_chunks --split dev
# Chỉ sau khi Round 1 official thỏa trigger:
python -m src.cli.build_passages --round2-all
python -m src.cli.eval_chunks --round2-all --retriever bm25 --split dev --official
# Sau khi 120/120 gold approved và official dev matrix hoàn tất:
python -m src.cli.lock_chunk_ablation --strategy B5
python -m src.cli.eval_chunks --strategy B5 --retriever bm25 --split test --official
```

Benchmark review vòng 2 nằm tại `data/08_eval/review_v2.csv`; bản compact chỉ gồm
query và evidence cần duyệt nằm tại `data/08_eval/review_v2_queries.csv`. Các record đã thay
query, gold hoặc required evidence được đưa về `review_status=draft`; record tốt từ
vòng trước vẫn giữ `approved`. Cột `distractor_node_ids` ghi các node gây nhiễu thật
cho category `hard_distractor`. Các cột `parent_contribution`, `shared_concepts`,
`discriminating_fact`, `lexical_overlap_ratio` và `longest_common_token_run` hỗ trợ
review đúng semantic contract của từng category.

Sau khi sửa review sheet, nhập lại và chạy quality gate bằng:

```powershell
python -m src.cli.benchmark_dataset --apply-review data/08_eval/review_v2.csv
python -m src.cli.benchmark_dataset --validate --require-approved
```

`--apply-review` đồng bộ cả query đã sửa nhưng không cho đổi `query_id`, split,
category hoặc document. Chỉ dùng `--official` sau khi toàn bộ 120 gold record có
`review_status=approved`. Test split là held-out và chỉ được chạy một lần sau khi
khóa strategy/config.

Các thay đổi benchmark có thể tái lập từ file curated revision:

```powershell
python -m src.cli.benchmark_dataset --apply-revisions data/08_eval/revisions_after_review_v1.yaml
```

### B6 dual-granularity one-shot

B6 dùng B1 làm Article scout và B4e làm fine evidence; không sinh passage corpus mới. Cấu hình RRF nằm trong `configs/retrieval.yaml` và được dùng nguyên vẹn cho BM25/Dense.

```powershell
python -m src.cli.eval_chunks --strategy B6 --retriever bm25 --split dev --official
python -m src.cli.eval_chunks --strategy B6 --retriever dense --split dev --official
python -m src.cli.b6_analysis --analyze
python -m src.cli.b6_analysis --export-review
# Điền 20 dòng review, sau đó:
python -m src.cli.b6_analysis --apply-review <reviewed.csv>
python -m src.cli.b6_analysis --finalize
# Chỉ sau finalize/strategy lock, chạy đúng winner và đúng một lần/retriever:
python -m src.cli.eval_chunks --strategy B6 --retriever bm25 --split test --official
python -m src.cli.eval_chunks --strategy B6 --retriever dense --split test --official
```

## PostgreSQL/pgvector shadow store

Frozen B6 vectors can be imported into an isolated PostgreSQL 18 + pgvector
container without changing the native PostgreSQL service or reopening held-out
evaluation. See `docs/postgres_pgvector.md` for setup, migration, validation,
four-stage shadow comparison, backup and restore-check commands.

The default runtime backend can be checked and queried with:

```powershell
python -m src.cli.retrieve --health
python -m src.cli.retrieve "Khoản 1 Điều 2 quy định nội dung gì?" --top-k 5 --compact
```

`--finalize` áp strict rule Recall@5 → MRR → EvidenceCoverage@5 → evidence tokens@5 → corrected duplicate-token ratio@5 → strategy ID, rồi khóa đúng một winner. Lệnh sẽ fail nếu review chưa đủ 20/20 approved hoặc có verdict `benchmark_issue`.
