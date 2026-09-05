# Traffic Law Parser

## Scope hiện tại

Hiện tại project tập trung xử lý trước **Luật 35/2024/QH15** và **Luật 36/2024/QH15** để làm bộ corpus chuẩn, sau đó mới mở rộng thêm các văn bản khác.

Mục tiêu chính là xây dựng một pipeline từ PDF → canonical legal tree → retrieval/evaluation, trong đó **cấu trúc pháp lý gốc được giữ làm nguồn sự thật (source of truth)**.

Pipeline tổng quát:

```text
registry
  → PDF inspect/render
  → native extraction
  → normalization
  → legal state machine
  → validation
  → validated corpus
  → legal tree / retrieval passages
  → benchmark & evaluation
```

## Legal tree thay cho chunking thông thường

Thay vì chỉ chia văn bản theo fixed-size chunk, project tổ chức văn bản theo cấu trúc phân cấp của văn bản pháp luật:

```text
Văn bản
  └── Chương
       └── Mục
            └── Điều
                 └── Khoản
                      └── Điểm
```

Cách tiếp cận này theo hướng **hierarchical legal tree**: quá trình retrieve có thể đi từ node ở cấp cao xuống node chi tiết hơn, thay vì coi toàn bộ văn bản như một tập chunk độc lập.

**Leaf node** là đơn vị thông tin chi tiết nhất, thường tương ứng với **Điểm** trong Điều/Khoản. Canonical node ID được giữ ổn định để làm reference/citation; các representation phục vụ retrieval được rebuild từ canonical corpus và không mutate nguồn dữ liệu gốc.

Project có tham khảo hướng tiếp cận về structural complexity của legal trees, trong đó cấu trúc phân cấp của văn bản được tận dụng như một phần của bài toán information retrieval.

## PDF parser pipeline

Phần PDF parser được triển khai theo pipeline nhiều bước:

1. **PDF inspect/render** để kiểm tra cấu trúc và hỗ trợ debug.
2. **Native extraction** bằng PyMuPDF để lấy text/block information.
3. **Normalization** để chuẩn hóa nội dung trước khi parse.
4. Nhận diện các marker pháp lý như **Chương, Mục, Điều, Khoản, Điểm**.
5. Dùng **state machine** để tái tạo lại đúng cấu trúc phân cấp.
6. Chạy validation trước khi publish corpus.

Build chỉ publish vào `data/05_validated` khi validation PASS; ERROR/FATAL đi vào quarantine. WARN cần manual review trước khi được approve.

## Đọc canonical corpus

Canonical corpus có thể được load và duyệt theo cấu trúc Điều/Khoản/Điểm, với mỗi node giữ thông tin nguồn gốc (trang, block) để trace lại văn bản gốc.

## Chunking và retrieval

Canonical legal tree vẫn là **nguồn sự thật**. Từ cùng corpus này, hệ thống chunking tạo ra các retrieval passages dạng rebuildable để benchmark nhiều strategy khác nhau.

Các strategy hiện có trong ablation gồm **B0-B5**, trong đó có:

- **Fixed window**: ví dụ 400 tokens với overlap 80 tokens.
- **Structural / hierarchical passages**: chia theo từng cấp như Điều/Khoản/Điểm và bổ sung context từ parent.
- Các strategy khác được rebuild từ cùng canonical tree để có thể so sánh công bằng.

Retrieval pipeline có thể sử dụng **BM25** hoặc **BGE-M3 dense retrieval**:

```text
validated canonical tree
  → B0-B5 retrieval passages
  → BM25 hoặc BGE-M3 dense retrieval
  → evidence expansion / structural processing
  → retrieval metrics
```

Round 2 và test split chỉ được chạy theo workflow benchmark đã khóa, sau khi Round 1 official thỏa trigger và 120/120 gold record đã được approve.

## Benchmark dataset

Project có một benchmark riêng để kiểm tra nhiều loại query, thay vì chỉ đánh giá một kiểu truy vấn.

Các category gồm:

- `exact_reference`
- `semantic_paraphrase`
- `point_specific`
- `multi_evidence`
- `hard_distractor`

Review vòng 2 nằm tại `data/08_eval/review_v2.csv`; bản compact gồm query và evidence cần duyệt nằm tại `data/08_eval/review_v2_queries.csv`.

Với `hard_distractor`, trường `distractor_node_ids` ghi các node gây nhiễu thực tế. Các trường `parent_contribution`, `shared_concepts`, `discriminating_fact`, `lexical_overlap_ratio` và `longest_common_token_run` hỗ trợ review đúng semantic contract của từng category.

Sau khi sửa review sheet, các thay đổi được đồng bộ lại vào dataset nhưng không cho đổi `query_id`, split, category hoặc document. Test split là held-out và chỉ chạy sau khi strategy/config đã được khóa.

Các benchmark thay đổi có thể tái lập từ curated revision file để đảm bảo lịch sử chỉnh sửa minh bạch.

## Metrics

Các strategy được đánh giá chủ yếu qua các nhóm metric về retrieval quality và evidence quality, gồm:

- **Recall@k**
- **MRR**
- **Evidence Coverage**
- **Evidence tokens**
- **Duplicate token ratio**
- **Parent/child coverage**
- **Structural coverage**

Regression snapshot được tạo tại `reports/<document_id>/regression_snapshot.jsonl` sau mỗi build.

## B6 – Hybrid / dual-granularity retrieval

Strategy **B6** được dùng để kết hợp retrieval ở hai mức độ chi tiết:

```text
B1 = Article scout
B4e = fine-grained evidence
```

B6 dùng **B1 làm Article scout** và **B4e làm fine evidence**, sau đó kết hợp kết quả theo RRF. Strategy này **không sinh passage corpus mới**; cấu hình RRF nằm trong `configs/retrieval.yaml` và được dùng nguyên vẹn cho cả BM25 và Dense.

Ý tưởng chính là không chỉ search ở một granularity duy nhất mà tìm đồng thời:

```text
Query
  ├── Article-level retrieval
  └── Clause/Point-level retrieval
           ↓
        RRF fusion
           ↓
      dedupe / evidence
```

## B7c – Rule-based routing

B7c bổ sung một lớp **rule-based routing** cho các query có legal reference rõ ràng.

Ví dụ:

```text
"Khoản 1 Điều 5"
```

thay vì đưa query qua semantic search như một query tự do, hệ thống có thể:

```text
Query
  ↓
Detect explicit reference
  ↓
Resolve canonical node trực tiếp
  ↓
Build evidence
  ↓
Return result
```

Như vậy, các query reference rõ ràng có thể được route trực tiếp tới đúng **Điều/Khoản/Điểm**, giảm chi phí retrieval không cần thiết và tận dụng cấu trúc canonical legal tree.

Những query không resolve được bằng rule-based routing sẽ tiếp tục đi vào retrieval/fallback path.

## PostgreSQL / pgvector shadow store

Frozen B6 vectors có thể được import vào một PostgreSQL 18 + pgvector container độc lập mà không thay đổi native PostgreSQL service hoặc mở lại held-out evaluation.

Chi tiết setup, migration, validation, shadow comparison và backup/restore nằm tại `docs/postgres_pgvector.md`.

## Test

Gold suite hiện có 25 case cho mỗi luật, bao phủ Điều/Khoản/Điểm, `đ)`, node đa trang, header và fidelity.

## Benchmark workflow / quality gate

Quy trình benchmark được thiết kế để tách rõ:

```text
parser correctness
    ↓
canonical legal tree
    ↓
passage/chunking ablation
    ↓
retrieval benchmark
    ↓
strategy analysis
    ↓
strategy lock
    ↓
held-out test
```

Khi finalize strategy, hệ thống áp strict rule theo thứ tự:

```text
Recall@5
→ MRR
→ EvidenceCoverage@5
→ evidence tokens@5
→ corrected duplicate-token ratio@5
→ strategy ID
```

Quá trình finalize sẽ fail nếu review chưa đủ 20/20 approved hoặc có verdict `benchmark_issue`.
