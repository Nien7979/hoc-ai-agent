# Nghiên cứu repo "muse-protocol" — 23/11/2026

**CẢNH BÁO NHẦM LẪN: có HAI repo cùng tên `muse-protocol`, hoàn toàn khác nhau.**
Đừng nhầm lẫn khi tìm, khi clone, khi đọc bài viết về chúng.

Nguồn chính đã đọc trực tiếp (chỉ ĐỌC file, KHÔNG chạy code nào trong repo):
- `muse-to-muse/` = clone của https://github.com/agathathemuse/muse-protocol (README, SPEC.md, SKILL.md)
- `muse-ai-signup/` = clone của https://github.com/duwqijlk/muse-protocol (README tiếng Trung)
- Bài phân tích: https://explainx.ai/blog/muse-protocol-github-meta-muse-signup-http-client-tos-risks-2026

---

## 1. agathathemuse/muse-protocol — giao thức để các Muse nói chuyện với nhau

**Đây là repo "Muse-to-Muse". Không liên quan gì tới việc đăng ký tài khoản Meta.**

### Nó là gì
- Giao thức mở `muse-protocol/1`: cho hai Muse cá nhân (mỗi Muse phục vụ một con người) giới thiệu bản thân và trao đổi tin nhắn qua Internet, không cần đưa API key cho nhau, không mở firewall, không qua trung gian.
- Tác giả ví: "email nếu email được thiết kế cho AI" — phát hiện công khai, phong bì có kiểu, mã hoá đầu-cuối tuỳ chọn, trả lời bằng polling thay vì inbox.
- Không registry, không server trung tâm: Muse nào muốn tham gia chỉ cần publish một manifest.

### Cách hoạt động (4 bước)
1. **Discovery:** Muse publish manifest JSON tại `/.well-known/muse-protocol.json` trên domain của chủ nó (tên Muse, phục vụ ai, endpoint nhận tin, URL poll trả lời, khoá mã hoá, khoá ký v1.1).
2. **Introduce:** Muse khách POST một phong bì JSON tới endpoint; server kiểm tra, xếp vào hàng đợi, trả về ticket.
3. **Poll:** Muse khách poll `reply.php?ticket=…` cho tới khi Muse chủ viết trả lời (`status: "ready"`).
4. **Message:** hai bên trao đổi phong bì `type: "message"` trong cùng thread.

### Mã hoá (tuỳ chọn, NaCl)
- `pwhash-xchacha20poly1305`: khoá = Argon2id(passphrase, salt); passphrase do hai con người trao đổi ngoài kênh (gặp mặt, điện thoại, giấy). Server chỉ lưu ciphertext.
- `box-x25519-xsalsa20poly1305`: mã hoá bằng public key trong manifest người nhận, không cần bí mật chung.

### Định danh: fingerprint `musefp/2`
- Chỉ là heuristic (dấu hiệu gợi ý), KHÔNG phải bằng chứng: các marker thô về VM (os_family, vm, container dạng boolean), SHA-256 fileproof của file sản phẩm (không kèm đường dẫn tuyệt đối). Spec nói thẳng mọi marker đều giả mạo được.
- Ký fingerprint bằng Ed25519 (`--sign`) thì gắn được attestation với khoá dài hạn — chống copy/replay (kèm `issued_at`, hiệu lực 24h).

### v1.1: chữ ký, proof-of-work, tier xác thực
- Envelope có thể mang chữ ký Ed25519 (`sig` + `kid`); manifest publish `signing_key` + `endorsements` (web-of-trust, không cần CA).
- `introduce` nên kèm PoW 20 bit (SHA-256, đào ~1 giây) — spam 10.000 cái tốn 10.000 giây, thành không kinh tế.
- Tier: `legacy` (không chứng minh gì, 3/ngày/IP) / `knock` (PoW, 5/ngày) / `signed` (chữ ký đúng, 20/ngày) / `inner-circle` (ký + passphrase, 50/ngày). **Không tier nào được auto-trust — tất cả vào hàng đợi kiểm duyệt, không gì tới tay con người nếu chưa qua duyệt.**

### Công cụ (Python 3, `muse-protocol/bin/`, cần `pynacl` nếu mã hoá)
`mp-discover`, `mp-send` (`--encrypt pwhash|box`, `--passphrase`, `--sign`, `--dry-run`), `mp-poll`, `mp-keygen` (X25519), `mp-sigkeygen` (Ed25519), `mp-check`, `mp-open`, `fingerprint` / `verify`, `new_passphrase`. Thư viện chung `mplib.py`. Đóng gói sẵn dạng skill: `muse-protocol/SKILL.md`.
- **Server tham chiếu:** PHP (`server/introduce.php`, `reply.php`, `lib.php`) — thả lên bất kỳ shared host nào (PHP 7.4+, sodium extension), queue nằm NGOÀI web root, chmod 700, payload cap 64 KB.
- **Ví dụ live:** Agatha (Muse của Luke) — manifest https://www.lukehurd.com/muse/.well-known/muse-protocol.json ; endpoint https://www.lukehurd.com/muse/api/muses/introduce.php

### Liên quan tới Hoai — RẤT CAO
- Hoai chính là "một Muse phục vụ một con người" (Javis phục vụ Lê Niên). Đây là lớp giao tiếp đối ngoại cho Văn Phòng AI: 14 phòng ban là các Muse, sau này có thể nói chuyện với Muse của người khác.
- **dammedien.com (khi kích hoạt):** publish manifest tại `dammedien.com/.well-known/muse-protocol.json` + deploy 3 file PHP → Javis có "cửa trước" công khai để Muse khác gõ cửa.
- Thử nghiệm rẻ nhất: `--dry-run` gửi introduce tới Agatha, không cần server riêng (Scenario A chỉ cần passphrase + endpoint bên kia).
- Không có rủi ro ToS kiểu Meta — đây là giao thức mở, không tự động hoá tài khoản của ai.

---

## 2. duwqijlk/muse-protocol — đăng ký muse.ai + đổi invite code bằng HTTP thuần

**Đây là repo đang TRENDING giữa tháng 11/2026 (~1.800 sao, lên 1.165 sao trong 2 ngày, GitTrend #49 ngày 15/11). Không liên quan gì tới repo số 1.**

### Nó là gì
- Python thuần HTTP (không cần trình duyệt) thực hiện toàn bộ flow đăng ký tài khoản Meta Muse.ai + đổi mã mời, ghi lại từ bắt gói tin (packet capture), đối chiếu từng field.
- Package `musepy/`: `client.py` (đăng ký + đổi mã), `runner.py` (chạy batch nhiều tài khoản, sticky proxy, ngân sách), `otp.py` (poll mã OTP,幂等 — chỉ lấy mã mới nhất sau lúc send), `falco.py` / `caa.py` (telemetry tuỳ chọn, không ảnh hưởng đăng ký).

### Flow HTTP đã map (theo README)
```
GET  /                                   lấy cookie ban đầu
POST /api/auth/native/restart            → 201 {csrf_token} — body PHẢI là {"resume": false}, không phải {}
POST /api/auth/native/send-otp           → 201
POST /api/auth/native/confirm-otp        → 200 {registration_state:"completion_required", tos:{content_version}}
                                          — BẮT BUỘC kèm "supports_muse_link_transparency": true
POST /api/auth/native/register-completion→ 200 {auth_meta_session:{frl_access_token, frl_account_id}}
                                          — first_name = tên (given), last_name = họ (family), NGƯỢC trực giác;
                                            sinh nhật <18 tuổi → 403
POST auth.meta.com/hatch/session/bootstrap/ → 302…→ muse.ai 200 HTML (BẮT BUỘC — không bootstrap thì mọi API hatch sau đó đều 403)
     └ parse disclosure_text/disclosure_version động (hiện tại HATCH_ROW_v0) + trạng thái resolvedDestination/isGeoGated/waitlist
POST /api/hatch/accept-tos               → 200 — body DÙNG ĐÚNG văn bản disclosure server trả về (không hard-code), gọi 1 lần
GET  /api/hatch/invite                   → 200 = tài khoản hợp lệ
POST /api/hatch/vm/wake                  → 200 — BẮT BUỘC (wake_trigger=mount + open_id + away_ms=0; vmId trích từ HTML trang đã login)
                                          — thiếu bước này thì đổi mã hợp lệ cũng báo sai "invalid / mã không tồn tại"
POST /api/hatch/stateful-sharing         → 200
POST /api/hatch/invite-code/redeem       → 200 {redemptionStatus:"redeemed"}
POST /api/hatch/invite                   → 200 {code} — lấy lại mã của chính mình
```
- Lý do đổi mã thất bại: `redeemed` (ok) / `already_redeemed` (tài khoản đã đổi 1 mã rồi) / `used_up` (mã đạt giới hạn 30 lượt) / `invalid` (mã không tồn tại HOẶC chưa gọi `vm/wake`).
- **Cookie:** `hatch_sess` (~886 ký tự) do `register-completion` trả về trong header — đây là credential cho mọi API hatch sau đó. `hatch_vml` trong lúc đăng ký luôn bị clear (Max-Age=0), không bắt buộc.
- Batch: worker độc chiếm 1 dòng proxy (sticky), giới hạn thành công/proxy mặc định 6, lỗi mạng trước khi gửi OTP thì đổi proxy retry tối đa 3 lần; lỗi nghiệp vụ / đã gửi OTP / region_waitlist thì DỪNG CẢ BATCH. Ngân sách = số email khác nhau, thất bại không bù số.
- README ghi thêm: dùng node JP thì "không cần gắn thẻ, qua thẳng" — claim CHƯA được kiểm chứng độc lập; explainx.ai không khuyến nghị dùng chiêu đổi vùng để né kiểm tra thanh toán.

### Liên quan tới Hoai — HIỂU CƠ CHẾ, KHÔNG DÙNG ĐỂ ĐĂNG KÝ HÀNG LOẠT
- **Giá trị thật:** tài liệu này GIẢI THÍCH cơ chế auth của muse.ai mà muse2api đang dựa vào — `hatch_sess` là credential chính, `accept-tos` cần disclosure động, `vm/wake` cần vmId, bootstrap là bắt buộc. Hiểu được flow này thì hiểu vì sao cookie import của muse2api cần đúng 2 cookie tối thiểu (`hatch_sess` + `hatch_native_auth_device`) và vì sao gateway hay flapping.
- **RỦI RO — nói thẳng:**
  - Tự động hoá flow OTP/đăng ký có thể vi phạm Điều khoản Meta, dù kỹ thuật làm được.
  - Xoay proxy để né cổng địa lý (`isGeoGated`, `waitlist`) có thể bị coi là lách luật tuỳ pháp luật/hợp đồng.
  - Chạy batch nhiều tài khoản = đúng nhóm rủi ro khoá tài khoản mà Hoai đã chấp nhận cho muse2api — nhưng ở mức cao hơn (đăng ký hàng loạt). Repo tự khai chỉ dùng cho "nghiên cứu cá nhân, phi thương mại", KHÔNG có giấy phép mã nguồn mở.
  - **Kết luận: chỉ đọc để hiểu, KHÔNG chạy code này, KHÔNG đăng ký tài khoản hàng loạt, KHÔNG dùng proxy JP để né kiểm tra.** Hoai đã có tài khoản Muse của mình, không cần thêm.
- Lưu ý phân biệt: "Meta Personal Agent Protocol" (Meta + Sierra công bố 6/10/2026, agent ↔ doanh nghiệp) là thứ KHÁC, không phải repo này.

---

## Đề xuất bước tiếp theo (chờ Hoai chọn)
1. **Thử khô với Agatha (rẻ, an toàn):** `mp-discover` + `mp-send --dry-run` tới manifest của Agatha để xem phong bì introduce trông như thế nào — không gửi thật nếu Hoai chưa đồng ý.
2. **Chuẩn bị manifest cho dammedien.com:** khi domain kích hoạt, publish manifest + deploy endpoint PHP để Javis có cửa trước công khai.
3. **Chỉ lưu ghi chú này:** không làm gì thêm với cả hai repo.
