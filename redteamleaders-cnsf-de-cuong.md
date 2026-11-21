# Certified Network Security Fundamentals (CNSF) — Đề cương ôn tập

- Nguồn: trang kỳ thi RedTeamLeaders — https://courses.redteamleaders.com/exams/2be49538-14af-417b-9ac6-c68d84d23955
- Tài khoản: lhnien95@gmail.com (Niên Lê Hữu)
- Trạng thái: **đã đăng ký từ 21/11/2026 (một trong 6 kỳ thi đăng ký bằng mã NEXTSTEP100, 0đ, không thẻ); còn 1 lượt thi, chưa từng thi. Chưa bấm "Start Exam" — kỳ thi để Chủ tịch quyết định sau.**
- Định dạng kỳ thi: 80 câu — 60 trắc nghiệm + 20 câu tự luận (free-text); 2 giờ; điểm đậu 75%; online, tính giờ; độ khó Beginners; ngôn ngữ English.
- Mô tả: chứng nhận kiến thức nền tảng về an ninh mạng — mạng hoạt động thế nào, các giao thức kẻ tấn công hay lợi dụng, và các biện pháp phòng thủ.
- Lưu ý: trang này chỉ là trang đăng ký kỳ thi, KHÔNG có bài học/module/video. Nội dung học duy nhất là mục "Study Resources" (đề cương bên dưới). Hai file PDF "The Network Security Fundamentals Study Guide" và "Exam Blueprint" trên trang đã **404 thật trên server (kiểm tra 21/11 ~22:03)** — không đọc được toàn văn.

## A. Nền tảng mạng (Networking fundamentals)

- **Mô hình OSI 7 tầng:** Physical (vật lý: cáp, tín hiệu), Data Link (MAC, switch, frame), Network (IP, routing), Transport (TCP/UDP, cổng), Session, Presentation (mã hóa, nén), Application (HTTP, DNS…). Nhớ thứ tự và ví dụ giao thức/thiết bị ở mỗi tầng — dạng câu hỏi trắc nghiệm hay gặp.
- **Bộ giao thức TCP/IP 4 tầng:** Link, Internet, Transport, Application; tương ứng rút gọn của OSI.
- **Subnetting:** chia mạng bằng subnet mask, tính số host, địa chỉ network/broadcast; ví dụ /24 = 256 địa chỉ, 254 host dùng được. Cần thành thạo phép AND địa chỉ IP với mask.
- **Routing và Switching:** router chuyển gói giữa các mạng theo bảng định tuyến; switch chuyển frame trong mạng LAN theo địa chỉ MAC; phân biệt collision domain và broadcast domain.
- **VLAN:** chia LAN thành các mạng logic riêng để cô lập lưu lượng và giảm broadcast; trunk (802.1Q) mang nhiều VLAN giữa các switch.
- **NAT:** dịch địa chỉ riêng (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) sang địa chỉ công khi ra Internet; PAT dùng cổng để nhiều máy dùng chung một IP công.
- **DNS:** phân giải tên miền thành IP; các bản ghi A, AAAA, CNAME, MX, NS; phân biệt DNS server đệ quy và authoritative.
- **DHCP:** cấp IP tự động; quy trình DORA (Discover – Offer – Request – Acknowledge); rogue DHCP là DHCP giả mạo cấp IP/gateway độc hại.
- **ARP:** phân giải IP thành MAC trong LAN; bảng ARP; ARP spoofing lợi dụng việc ARP không xác thực.
- **ICMP:** ping, traceroute, báo lỗi mạng; ICMP cũng bị lợi dụng trong quét mạng và tấn công.
- **TCP và UDP:** TCP có kết nối, bắt tay 3 bước SYN–SYN/ACK–ACK, đảm bảo thứ tự và truyền lại khi mất gói — dùng cho web, email, file. UDP không kết nối, nhanh, không đảm bảo — dùng cho DNS, video, VoIP. Nhớ số cổng phổ biến: 21 FTP, 22 SSH, 23 Telnet, 25 SMTP, 53 DNS, 80 HTTP, 110 POP3, 143 IMAP, 443 HTTPS, 3389 RDP.
- **HTTP/HTTPS:** HTTP truyền dạng rõ (plain text) nên bị nghe lén; HTTPS dùng TLS để mã hóa, xác thực server bằng chứng chỉ.

## B. Kiến thức cơ bản về an ninh mạng (Network security basics)

- **Firewall:** lọc lưu lượng theo IP, cổng, giao thức và trạng thái (stateful); đặt ở biên mạng; quy tắc mặc định nên là chặn tất cả, chỉ cho phép cái cần thiết (deny by default).
- **IDS/IPS:** IDS phát hiện và cảnh báo; IPS phát hiện và chặn ngay; phân biệt phát hiện theo chữ ký (signature) và theo bất thường (anomaly); đặt ở vị trí span/tap hay inline.
- **VPN:** tạo kênh mã hóa qua mạng công cộng; site-to-site nối hai văn phòng, remote-access cho người dùng từ xa; phân biệt VPN mã hóa toàn bộ lưu lượng với split tunnel.
- **Segmentation (phân đoạn mạng):** chia mạng thành vùng (ví dụ DMZ cho server công khai, mạng nội bộ, mạng quản trị) để hạn chế lan truyền khi một vùng bị xâm nhập.
- **Access Control Lists (ACL):** danh sách cho phép/chặn lưu lượng theo nguồn, đích, cổng, giao thức; áp trên router/firewall; chú ý thứ tự quy tắc và quy tắc ngầm cuối cùng.
- **Zero trust:** "không tin tưởng mặc định" — xác minh mọi truy cập, dù trong hay ngoài mạng; nguyên tắc: xác thực liên tục, quyền tối thiểu, giả định đã bị xâm nhập.
- **Secure network design:** thiết kế phòng thủ theo chiều sâu (defense in depth), tách vùng, dự phòng, ghi nhật ký tập trung, cập nhật và quản lý cấu hình.

## C. Các tấn công thường gặp (Common attacks)

- **ARP spoofing:** kẻ tấn công gửi ARP giả, gán MAC của mình cho IP của gateway/nạn nhân để chặn và chuyển hướng lưu lượng.
- **DNS spoofing:** trả lời DNS giả để đưa nạn nhân tới trang độc hại; phòng bằng DNSSEC, dùng DNS đáng tin, giám sát.
- **Port scanning:** quét cổng mở để vẽ bản đồ dịch vụ (ví dụ Nmap); cổng mở không cần thiết là bề mặt tấn công — cần đóng hoặc lọc.
- **Brute force attacks:** thử mật khẩu liên tục; phòng bằng mật khẩu mạnh, khóa tài khoản sau nhiều lần sai, MFA, giới hạn tốc độ đăng nhập.
- **Packet sniffing:** bắt gói tin trên mạng, đặc biệt nguy hiểm với giao thức không mã hóa (HTTP, FTP, Telnet); phòng bằng mã hóa và switch thay cho hub.
- **Man-in-the-middle (MITM):** kẻ tấn công đứng giữa hai bên, nghe lén hoặc sửa dữ liệu; thường kết hợp ARP spoofing/DNS spoofing/Wi-Fi giả.
- **Denial-of-service (DoS):** làm quá tải dịch vụ/mạng khiến người dùng hợp lệ không truy cập được; DDoS dùng nhiều nguồn; phòng bằng giới hạn tốc độ, lọc lưu lượng, CDN/WAF.
- **Rogue DHCP:** máy chủ DHCP giả cấp gateway/DNS độc hại để chuyển hướng lưu lượng; phòng bằng DHCP snooping trên switch.
- **Credential interception:** đánh cắp tài khoản/mật khẩu qua nghe lén, lừa đảo (phishing), keylogger hoặc giao thức không mã hóa; hậu quả là chiếm quyền truy cập.

## D. Các biện pháp phòng thủ (Defensive controls)

- **Network monitoring:** giám sát lưu lượng và thiết bị liên tục để phát hiện bất thường sớm.
- **Logging:** ghi nhật ký tập trung (syslog/SIEM), lưu đủ lâu, đồng bộ thời gian, rà soát log định kỳ.
- **Hardening:** tắt dịch vụ/cổng không dùng, đổi mật khẩu mặc định, vô hiệu hóa tài khoản không cần thiết, cấu hình an toàn.
- **Patching:** vá lỗ hổng hệ điều hành, firmware, ứng dụng kịp thời; ưu tiên lỗ hổng đang bị khai thác.
- **Least privilege (quyền tối thiểu):** mỗi người dùng/dịch vụ chỉ có đúng quyền cần để làm việc, không hơn.
- **MFA:** xác thực đa yếu tố (mật khẩu + mã ứng dụng/khóa bảo mật) để giảm rủi ro khi mật khẩu bị lộ.
- **Secure configuration:** cấu hình chuẩn, kiểm tra định kỳ, quản lý thay đổi, sao lưu cấu hình.
- **Encryption:** mã hóa dữ liệu khi truyền (TLS/VPN) và khi lưu trữ; quản lý khóa an toàn.
- **Incident response basics:** các bước cơ bản khi có sự cố — phát hiện, khoanh vùng, loại bỏ, khôi phục, rút kinh nghiệm; giữ bằng chứng và báo cáo theo quy trình.

## E. Công cụ thực hành (Hands-on tools)

- **Wireshark:** bắt và phân tích gói tin, lọc theo giao thức/IP/cổng, đọc được bắt tay TCP, truy vấn DNS, nội dung HTTP chưa mã hóa.
- **Nmap:** quét host và cổng, xác định dịch vụ và hệ điều hành, ví dụ `nmap -sV` để dò phiên bản dịch vụ.
- **Netcat:** kiểm tra kết nối TCP/UDP, gửi/nhận dữ liệu, kiểm tra cổng mở, hỗ trợ truyền file và shell đơn giản.
- **tcpdump:** bắt gói tin bằng dòng lệnh trên Linux, hữu ích khi không có giao diện đồ họa.
- **traceroute:** xác định đường đi của gói tin qua các router, giúp chẩn đoán lỗi mạng.
- **ping:** kiểm tra host có phản hồi ICMP không và đo độ trễ; bị chặn ICMP không có nghĩa là host chết.
- **nslookup/dig:** tra cứu DNS, kiểm tra bản ghi và chẩn đoán lỗi phân giải tên.
- **Basic firewall rules:** viết quy tắc cho phép/chặn theo IP, cổng, giao thức; luôn kiểm tra thứ tự và kiểm thử sau khi áp dụng.

## F. Luyện tập theo đề cương (Recommended practice)

- Phân tích packet capture (file .pcap): xác định giao thức, nguồn/đích, nội dung đáng ngờ.
- Nhận diện giao thức từ lưu lượng: HTTP, DNS, TLS, ARP, ICMP…
- Đọc kết quả quét Nmap: cổng mở/đóng/lọc, dịch vụ đang chạy, suy ra rủi ro.
- Nhận biết lưu lượng đáng ngờ: quét cổng, ARP bất thường, DNS lạ, lưu lượng tới IP không rõ, dữ liệu truyền đi bất thường.
- Giải thích bước giảm thiểu (mitigation) cho từng tình huống: chặn, vá, mã hóa, phân đoạn, giám sát, đào tạo người dùng.
- Tài liệu tham khảo gợi ý trên trang: Cisco Networking Basics; CompTIA Network+ study material; CompTIA Security+ network security sections; Wireshark official documentation; Nmap Network Scanning guide; OWASP testing concepts for web/network exposure; MITRE ATT&CK Enterprise: Network-based techniques; CISA cybersecurity fundamentals guidance.
