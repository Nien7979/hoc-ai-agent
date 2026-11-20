# Bài đã học: Microsoft AI Agents for Beginners — Phần 2c (Bài 14–18) — HOÀN THÀNH KHÓA

- Nguồn: Microsoft, "AI Agents for Beginners" (MIT), repo https://github.com/microsoft/ai-agents-for-beginners — đã fork về https://github.com/Nien7979/ai-agents-for-beginners và clone về máy.
- Ngày học: 20/11/2026 (đêm). Học tiếp sau phần 2b (bài 09–13).
- **Cả khóa 18 bài (00–18) đã học xong**: phần 1 (01–05), phần 2a (06–08), phần 2b (09–13), phần 2c (14–18). Bài 00 là setup môi trường (đã đọc qua AGENTS.md của repo).

## Bài 14 — Exploring Microsoft Agent Framework (MAF)

- **MAF** = framework thống nhất của Microsoft để build agent. Orchestration: sequential, concurrent, group chat, handoff, **magnetic** (manager agent tạo/sửa task list và điều phối subagent).
- Production features: **observability** (OpenTelemetry, Foundry dashboards), **security** (host trên Foundry: RBAC, private data, content safety), **durability** (thread/workflow pause, resume, recover), **control** (human-in-the-loop approval).
- Interop: cloud-agnostic (container, on-prem, nhiều cloud), provider-agnostic (Azure OpenAI, OpenAI, MiniMax qua API tương thích OpenAI, remote agent qua A2A), chuẩn mở (A2A, MCP), plugins/connectors (Fabric, SharePoint, Pinecone, Qdrant).
- **Core concepts**:
  - **Agents**: tạo bằng chat client + instructions + name (`create_agent`); chạy bằng `.run` / `.run_stream`; tool khai báo lúc tạo agent hoặc lúc chạy (chỉ cho lần chạy đó).
  - **Threads**: `get_new_thread()` cho hội thoại nhiều lượt; serialize/deserialize để lưu và khôi phục sau.
  - **Middleware**: function middleware (chạy giữa agent và tool — vd log function call) và chat middleware (chạy giữa agent và LLM — thấy được messages gửi đi); có pre/post processing, `next` để gọi bước tiếp theo.
  - **Memory**: in-memory (thread trong lúc chạy), persistent messages (`chat_message_store_factory`), dynamic memory (Mem0Provider qua `context_providers`, thêm vào context trước khi chạy).
  - **Observability**: `get_tracer()` / `get_meter()` của OpenTelemetry.
- **Workflows**: executor (nhận input, làm việc, trả output), edge (direct, conditional, switch-case, fan-out, fan-in), event (WorkflowStarted/Output/Error, ExecutorInvoke/Complete, RequestInfo). Cho phép orchestration multi-agent + checkpointing (lưu trạng thái workflow).
- **Advanced patterns**: middleware composition (chuỗi log + auth + rate-limit), workflow checkpointing (lưu/khôi phục tiến trình dài), dynamic tool selection (RAG trên mô tả tool, chỉ đưa tool liên quan), multi-agent handoff (edge + conditional routing).
- **Host LangChain/LangGraph agent trên Foundry**: package `langchain-azure-ai[hosting]`; `ResponsesHostServer` (endpoint `/responses`, OpenAI-compatible, mặc định khuyên dùng) vs `InvocationsHostServer` (`/invocations`, JSON tùy chỉnh/webhook). Deploy bằng Azure Developer CLI (`azd ai agent init/run/provision/deploy`); cần role Foundry Project Manager. Hỗ trợ conversation (`previous_response_id`), human-in-the-loop qua LangGraph `interrupt()` (hiện thành `function_call`/`mcp_approval_request`).

## Bài 15 — Building Computer Use Agents (CUA)

- **CUA** = agent dùng trình duyệt như người: mở browser, nhìn trang, làm hành động tiếp theo tốt nhất.
- Stack demo: **Browser-Use** (điều hướng bằng AI) + **Playwright + CDP** (điều khiển browser) + Azure OpenAI vision (hiểu trang) + Pydantic (trích dữ liệu có cấu trúc). Ví dụ: tìm Airbnb ở Stockholm, trích giá, chọn chỗ rẻ nhất.
- **Hybrid workflow**: Chrome chạy với CDP để Playwright và Browser-Use dùng chung một session; agent lo navigation động (mở trang, tắt popup, tìm kiếm), chuyển sang direct control khi tương tác đã predictable; Pydantic schema trích title/giá/rating/URL; code Python so sánh và chọn rẻ nhất.
- **Agent vs Actor**:
  | Tình huống | Agent | Actor |
  |---|---|---|
  | Layout động, hay đổi | Có (AI thích nghi) | Không (selector cứng dễ gãy) |
  | Cấu trúc đã biết | Không (chậm hơn) | Có (nhanh, chính xác) |
  | Tìm element | Có (ngôn ngữ tự nhiên) | Không (cần selector chính xác) |
  | Timing control | Không (khó đoán) | Có (chủ động wait/retry) |
  | Workflow phức tạp | Có (xử lý UI bất ngờ) | Không (cần branch tường minh) |
- **Best practices**: bắt đầu bằng agent để khám phá; chuyển sang direct control khi đã predictable; structured output để dữ liệu validate được; delay chiến lược sau action đổi UI; chụp screenshot khi debug; thiết kế fallback cho popup/layout đổi; blend agent + actor.
- **7 safety guardrail cho browser agent**:
  1. Scope môi trường duyệt (profile/sandbox riêng, giới hạn domain cần thiết).
  2. Tách observation khỏi action: tìm/đọc/trích xuất thì tự làm; submit form, gửi tin, đặt vé, mua hàng, xóa dữ liệu, đổi setting phải có approval tường minh.
  3. Không đưa secret (password, payment, cookie, dữ liệu cá nhân thô) vào prompt/trace; để user tự đăng nhập, redact field nhạy cảm khỏi log.
  4. Coi nội dung trang là **untrusted input**: trang có thể chứa lệnh cho agent — bỏ qua mọi text bảo đổi mục tiêu, lộ dữ liệu, tắt safeguard, sang site khác.
  5. Deterministic check quanh bước rủi ro: verify URL, title, item đã chọn, giá, người nhận, tóm tắt hành động bằng code trước khi xin duyệt.
  6. Budget + stop conditions: giới hạn số action, retry, tab, phút; dừng khi trạng thái trang mơ hồ thay vì click tiếp.
  7. Chỉ lưu evidence hữu ích (tóm tắt action, timestamp, URL, mô tả element, screenshot ref) — không lưu nội dung nhạy cảm thừa.
- **Project Opal** (Microsoft 365 Copilot, Frontier early access): CUA cấp doanh nghiệp chạy trên Windows 365 Cloud PC, async nền, user có thể Take Control/Return Control bất cứ lúc nào. Áp dụng đúng các bài trước: human-in-the-loop (dừng xin credential/dữ liệu nhạy cảm, không nhập password/submit form khi chưa confirm), trust (môi trường cô lập, browser-only, dùng identity của user, log mọi action), planning/metacognition (lập plan trước, tự giám sát từng bước, dừng khi thấy đáng ngờ), Skills (hướng dẫn job lặp lại từ file `.md`, tái dùng qua nhiều conversation).

## Bài 16 — Deploying Scalable Agents with Microsoft Foundry

- **Prototype vs production**: model chỉ ~20% của agent production; 80% còn lại là operational skeleton (hosting, identity, state, failure handling, cost, quality, trust).
- **3 deployment pattern**: client-hosted (agent trong process của app — toàn quyền control, tự lo scaling/state), hosted agents (Foundry Agent Service — đăng ký agent thành resource, Foundry host reasoning loop, lưu thread, RBAC, content safety; app chỉ là thin client), agent workflows (nhiều agent + tool thành graph có control flow tường minh, approval node, durable checkpoint).
- **Agent lifecycle**: Create → Version → Evaluate offline (gate) → Deploy → Observe online → Collect failures → Improve → Retire. **Offline evaluation là gate, không phải việc làm sau**: version mới không ship nếu không qua ngưỡng đánh giá; online observability đưa failure thật về bổ sung offline test set.
- **Scaling**: stateless request handling (không giữ state user trong process memory — thread lưu ở Foundry thread store/memory service, instance nào cũng xử lý được request nào → scale ngang); model routing (request đơn giản → model nhỏ nhanh rẻ, request khó → model lớn; Foundry Model Router hoặc classifier tự làm); response caching (câu hỏi lặp lại trả từ cache, không gọi model); concurrency + backpressure (giới hạn concurrency, retry exponential backoff, fail gracefully).
- **Observability**: OTel trace + attribute (`customer.tier`, `routed.model`) để trả lời câu hỏi kinh doanh ("khách enterprise có bị route sang model nhỏ quá nhiều không?").
- **Cost**: 3 đòn bẩy theo thứ tự — right-size model (model nhỏ nhất mà pass eval gate), route by complexity, cache aggressively. Eval gate và cost control là một: eval cho biết quality floor, routing/caching giữ chi phí gần floor đó nhất.
- **Enterprise**: governance (managed identity least privilege — chỉ đọc knowledge base, scoped access ticketing API); human-in-the-loop (approval-required tools: agent đề xuất → dừng → người duyệt/từ chối → workflow tiếp tục); **MCP in production = untrusted boundary** (pin version, scoped identity, validate output, không đưa secret, rate-limit — đối xử như mọi dependency bên thứ ba).
- **Smoke tests**: deploy thành công chỉ chứng minh control plane nhận definition, không chứng minh agent trả lời được. Smoke test (AI Smoke Test GitHub Action, catalog `tests/lesson-16-smoke-tests.json`) chạy mỗi lần deploy, kiểm tra endpoint có trả lời không. Pyramid: smoke test (mỗi deploy) → offline eval (trước promotion) → online eval (liên tục).
- **Lab**: Contoso customer support agent đầy đủ — tool (tra cứu đơn, mở ticket), RAG (Azure AI Search + fallback in-memory), memory, model routing (classifier độ phức tạp), caching, human approval (refund trên ngưỡng), evaluation pipeline (gate), observability (OTel). Bài tập: hardening thành billing support agent (tool subscription/invoice/credit, RAG policy hoàn tiền/chu kỳ/hủy, eval set ≥8 case gồm 2 case trigger approval, báo cáo cost sau 10 query).

## Bài 17 — Creating Local AI Agents (Foundry Local + Qwen)

- **Vì sao chạy local**: privacy (code/tài liệu không rời máy), cost (không tính tiền token), offline (máy bay, cơ sở bảo mật, mất mạng vẫn chạy).
- **SLM trade-off**: mạnh ở task có cấu trúc/giới hạn (classification, extraction, summarization), **tool calling** (quyết định gọi function nào với tham số nào), iterate nhanh/rẻ/riêng tư; yếu ở suy luận mở nhiều bước và kiến thức thế giới rộng. Chiến lược thắng: **để SLM orchestrate, để tool làm việc nặng** — model không cần biết codebase, chỉ cần biết khi nào gọi `read_file`/`search_docs`.
- **Foundry Local**: runtime tải/quản lý/serve model ngay trên máy, expose **OpenAI-compatible endpoint** (thường `http://localhost:PORT/v1`) — OpenAI SDK và MAF OpenAI client chỉ cần đổi `base_url`; tự chọn build CPU/CUDA/NPU phù hợp hardware. Cài: `winget install Microsoft.FoundryLocal` (Windows) / brew (macOS); `foundry model run qwen2.5-7b`.
- **Qwen function calling**: nhiều SLM chat được nhưng tool call không tin cậy (sai định dạng); Qwen được train riêng cho function calling, cho tool call đúng cấu trúc ổn định — đó là thứ biến local chat model thành local *agent*.
- **Local RAG**: Chroma (embedded vector store, chạy in-process, lưu trên disk). Pipeline hoàn toàn local: embedding model local → vector local → retrieval local → SLM local. Cùng pattern Agentic RAG (bài 05), chỉ khác mọi component đều ở trên máy.
- **Local MCP**: MCP là transport, không phải dịch vụ cloud — server MCP chạy local qua `stdio`, expose tool cho agent. Không vì local mà tự an toàn: server chạy với quyền của user → scope những gì nó được chạm (1 project dir, không phải cả home) và validate output.
- **Hybrid**: sensitive/offline → local SLM; task đơn giản → local (nhanh, rẻ); suy luận sâu + dữ liệu không nhạy cảm → cloud; outage → fallback local (giảm chất lượng thay vì sập hẳn). Đây là model routing (bài 16) với máy mình là một trong các "model".
- **Lab**: local engineering assistant — tool `read_file` (có sandbox check: chỉ trong project root), `analyze_code`, RAG docs qua Chroma, MCP local (graceful skip nếu không cấu hình). RAM tối thiểu thực tế **8 GB**, 16 GB+ thoải mái; GPU/NPU tăng tốc nhưng không bắt buộc.

## Bài 18 — Securing AI Agents with Cryptographic Receipts

- **Vấn đề**: log thường ai có quyền file cũng sửa được; cloud logging thì auditor phải tin nhà cung cấp. Với workload chịu quy định (tài chính, y tế, EU AI Act), "tin tôi đi" là không đủ. Receipt = JSON ghi lại agent đã làm gì, **ký bằng Ed25519**, verify được độc lập.
- **3 tính chất**:
  1. **Signature**: ký bằng private key của agent gateway; ai có public key tương ứng verify được offline.
  2. **Canonical encoding**: serialize bằng JCS (RFC 8785) trước khi ký — cùng nội dung logic → cùng bytes → cùng signature.
  3. **Hash chaining**: `previous_receipt_hash` nối receipt này với receipt trước — xóa hoặc sắp xếp lại receipt làm vỡ chuỗi.
- **Receipt mẫu**: `type`, `agent_id`, `tool_name`, `tool_args_hash`, `result_hash`, `policy_id`, `timestamp`, `sequence`, `previous_receipt_hash`, `signature {alg: EdDSA, sig, public_key}`. Dùng hash thay vì raw args/result để receipt nhỏ, không lộ PII/dữ liệu kinh doanh (auditor đối chiếu hash với bản copy lưu riêng).
- **Receipt chứng minh 3 điều**: attribution (key này đã ký nội dung này), integrity (nội dung không đổi sau khi ký), ordering trong sequence đã verify.
- **Receipt KHÔNG chứng minh**: correctness (hành động đúng hay sai), policy compliance (policy có thực sự được đánh giá không), identity beyond key (key ≠ người), truthfulness of inputs (prompt bị thao túng thì receipt vẫn ghi lại hành động một cách trung thực), execution/real-world effect (receipt ghi claim, không chứng minh hiệu ứng ngoài đời), completeness (thiếu receipt có thể là bị chặn/mất/chưa instrument), trusted time (timestamp là assertion của signer, không phải thời gian đáng tin).
- **Bẫy quan trọng nhất**: `public_key` trong receipt KHÔNG phải key để verify. Attacker có thể sửa receipt, ký lại bằng key của mình, nhúng key đó vào — receipt tự verify hoàn hảo với chính nó. Verifier phải lấy key từ nguồn độc lập (JWK Set đã pin, registry) và verify với key đó.
- **Human approval receipts** (`human.approval.v1`): approver key ký toàn bộ action + digest; agent key ký claim mang cùng digest + `parent_approval_ref` (hash của approval). `verify_chain` đi qua cả hai artifact với **2 registry key riêng biệt** (approver keys vs agent keys). **Stale authority** vẫn bị từ chối dù signature còn hợp lệ: policy version đã đổi, approver key đã rotate khỏi registry, approval hết hạn. **Digest substitution** (approval thật nhưng cho action khác) cũng bị từ chối với lý do riêng.
- **Production checklist**: key vào Key Vault/KMS/HSM (không để trong source/plaintext); publish key qua JWK Set, relying party pin identity + rotation policy; anchor chain head ra transparency log (Sigstore Rekor, RFC 3161 timestamp authority); lưu receipt append-only (immutability policies); retention (mỗi receipt ~500 bytes; agent 10K call/ngày → ~1.8 GB/năm); document rõ những gì receipt KHÔNG cover.
- **Thư viện production**: IETF draft `draft-farley-acta-signed-receipts` + test vectors `agent-governance-testvectors`; Microsoft Agent Governance Toolkit (Tutorial 33, Cedar policy); `protect-mcp` / `@veritasacta/verify` (npm, wrap MCP server bằng audit trail chống giả mạo, có held-for-co-sign flow); `nobulex` (Python SDK, LangChain/CrewAI); EMILIA Protocol (exact-action enforcement, WebAuthn/quorum authority, one-time consumption ở executor boundary, indeterminate-outcome path).
- **Nâng cao**: selective disclosure (Merkle tree RFC 6962 — tiết lộ từng field cho từng auditor mà không lộ phần còn lại), receipt revocation (short-lived key + revocation list), bilateral/split-signature receipts (pre-execution authorization và post-execution result ký riêng), post-quantum migration (ML-DSA — chỉ đổi label `alg` là chưa đủ, cần cả key encoding, canonical bytes, verification, downgrade resistance, transition semantics).

## Áp dụng cho công việc của Niên và Hoai

- **MAF/Workflow/Middleware (bài 14) = khung cho hệ phòng ban**: Niên là orchestrator/manager, middleware = log + approval gate trước mỗi lần đăng/xóa; checkpointing = queue JSON + manifest cho phép resume sau reboot (VM đã reboot 7 lần ngày 19–20/11 — đúng nhu cầu này).
- **CUA safety guardrails (bài 15) = đúng cách Niên dùng browser**: tách observation khỏi action (chỉ check, không mua/đăng khi chưa duyệt), coi nội dung trang là untrusted (không làm theo lệnh trong trang — đúng quy tắc chống prompt injection), verify URL/giá trước khi báo Hoai, giới hạn số action/retry.
- **Deploy (bài 16)**: mọi cron/watchdog nên có eval/smoke check — verify artifact thật chứ không tin browser report (bài học subtitle V35); stateless — state nằm trong file/manifest chứ không trong RAM; routing theo độ khó — model rẻ cho việc đơn giản, model mạnh cho việc khó (đúng như xKiro vs fosta hiện tại); eval gate trước khi "ship" — Niên duyệt video trước khi đăng chính là eval gate.
- **Local agents (bài 17)**: Ollama/Qwen chạy local trên máy Hoai là hướng rẻ + riêng tư khi cần; hybrid — việc nhạy cảm (dữ liệu cá nhân, key) → local, việc khó → cloud; khi cloud sập thì fallback local thay vì đứng im.
- **Receipts (bài 18)**: cho hệ thống đăng video/tiền bạc — mỗi lần đăng/xóa/chuyển tiền nên có receipt ký (agent nào, tool nào, hash args/result, timestamp, chain) để sau này kiểm tra được; key phải pin độc lập, không nhúng key vào chính receipt. Và ranh giới quan trọng nhất của bài này cũng là bài học lớn nhất của Niên: **"receipts prove attribution, not correctness"** — log/report đẹp không chứng minh việc đúng, phải verify artifact thật.
