# Báo Cáo Kỹ Thuật FinOps — Tối Ưu Hóa Chi Phí GPU (NimbusAI)

> **Tác giả:** FinOps Engineering Lead  
> **Khóa học:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25  
> **Dự án:** NimbusAI GPU Cost Optimization & Sustainability

---

## 1. Baseline vs. Optimized: Bức Tranh Toàn Cảnh

| Chỉ Số Đo Lường | Baseline (Trước tối ưu) | Optimized (Sau tối ưu) | Mức Giảm / Tiết Kiệm |
|---|---|---|---|
| **Tổng chi phí hàng tháng** | **$27,133 / tháng** | **$14,626 / tháng** | **-$12,507 / tháng (46%)** |
| **Đơn giá suy luận ($\text{\$/1M-token}$)** | **$6.488 / 1M token** | **$1.126 / 1M token** | **-82.6% đơn giá** |
| **Chi phí suy luận hàng ngày** | $48.87 / ngày | $8.48 / ngày | -$40.39 / ngày |
| **Chi phí thuê phần cứng (Workloads)** | $25,667 / tháng | $15,627 / tháng | -$10,040 / tháng (39.1%) |
| **Phát thải carbon (Schedulable compute)** | 1,606.3 kg $\text{CO}_2\text{e}$ | 126.8 kg $\text{CO}_2\text{e}$ | **-92.1% lượng phát thải** |

### Nhận xét cốt lõi
Việc chuyển đổi hệ quy chiếu từ **$\text{\$/GPU-giờ}$** sang **$\text{\$/1M-token}$** đã hé lộ tiềm năng cắt giảm chi phí suy luận lên tới **82.6%**. Mặc dù tổng số token phục vụ giữ nguyên (hơn 7.5 triệu token/ngày), đơn giá thực tế giảm từ \$6.49 xuống \$1.13 nhờ tái cấu trúc luồng phục vụ.

---

## 2. Phân Tích Chuyên Sâu Từng Đòn Bẩy (FinOps Levers)

| Đòn Bẩy | Tiết Kiệm Hàng Tháng | Tỷ Trọng Đóng Góp | Đánh Giá Tác Động & Cơ Chế |
|---|---|---|---|
| **1. Purchasing (Spot / Reserved)** | **$10,040 / tháng** | **80.3%** | **Đóng góp lớn nhất về số tiền tuyệt đối.** Khai thác các job training/batch có thể gián đoạn chuyển sang Spot (-40% đến -60% giá), và ký cam kết 1-Year Reserved cho các cụm inference chạy liên tục (>55% duty cycle). |
| **2. Inference Optimization** | **$1,212 / tháng** | **9.7%** | **Hiệu quả tỷ lệ cao nhất (-82.6%).** Kết hợp Cascade routing (80% traffic sang model nhỏ), Prompt Caching (giảm 90% input lặp lại) và Batch API (giảm 50% cho tác vụ offline). |
| **3. Right-size Util-Lies** | **$655 / tháng** | **5.2%** | **Sửa lỗi lãng phí phần cứng.** Thay thế các GPU H100 bị nghẽn bộ nhớ/I/O sang A100/A10G tương ứng với thông lượng thực tế cần dùng. |
| **4. Kill Idle GPUs** | **$600 / tháng** | **4.8%** | **Quick win không rủi ro.** Tắt triệt để `gpu-h100-5` khi để không qua đêm (00:00 - 08:00, utilization < 10%). |

---

## 3. Bản Chất Của "GPU-Util Lie" & Tác Động Tài Chính

### Cơ chế kỹ thuật
`nvidia-smi` chỉ lấy mẫu thời gian mà bộ xử lý GPU có lệnh đang chạy trên SM (Streaming Multiprocessor active clock), **không đo lường mật độ tính toán thực tế (FLOPs)**.
* Khi kernel bị nghẽn bộ nhớ (memory stall), chờ dữ liệu chuyển từ Host qua PCIe, hoặc khởi chạy kernel quá nhỏ (launch overhead), xung nhịp GPU vẫn được đánh dấu là 100% bận rộn.
* Trường hợp điển hình: **`gpu-h100-4`** đạt **98.2% GPU-Util** nhưng **MFU chỉ đạt 0.194 (19.4%)**.
* Trong khi đó, các GPU H100 được tối ưu hóa tốt (`gpu-h100-0` đến `gpu-h100-3`) đạt MFU từ 40% đến 43%.

### Tác động tài chính
Doanh nghiệp đang trả trọn vẹn \$2.50/giờ (\$1,800/tháng) cho một GPU H100 nhưng chỉ thu về năng lực tính toán tương đương một GPU A100 (\$1.79/giờ) hoặc thậm chí A10G (\$1.00/giờ). Cứ mỗi máy chủ bị "Util Lie", NimbusAI thất thoát hơn **$500 – $1,000 / tháng**.

---

## 4. Chi Tiết 5 Phần Mở Rộng Đã Triển Khai ("Your Turn")

### Extension 1: Ma Trận Ra Quyết Định Mua Sắm (Purchasing Matrix)
* **Vị trí code:** `finops/pricing.py` (`recommend_tier_matrix`) & `missions/m3_purchasing.py`.
* **Cải tiến:** Đưa thêm 3 yếu tố động vào thuật toán:
  1. *Tỷ lệ gián đoạn (preemption risk)* theo kiến trúc GPU (H100: 3%, A10G: 8%). Nếu rủi ro > 25%, từ chối dùng spot.
  2. *Thời hạn công việc (job horizon)* so với vòng đời phần cứng. Với các GPU thế hệ mới phát triển nhanh (H100, B200), ưu tiên cam kết **1-Year Reserved** thay vì 3 năm để phòng ngừa rủi ro lỗi thời công nghệ.
  3. Điểm hòa vốn: $55\%$ cho 3-year (45% discount) và $80\%$ cho 1-year (20% discount).
* **Kết quả:** Phân loại chính xác 5 job sang Spot (tiết kiệm \$8,242/tháng sau khi trừ chi phí checkpoint overhead & rework) và 3 job suy luận 24/7 sang 1-Year Reserved (tiết kiệm \$4,264/tháng).

### Extension 2: Right-Sizing Theo MBU & Kinh Tế Học VRAM
* **Vị trí code:** `missions/m1_efficiency_audit.py` (`analyze_mbu_rightsizing`).
* **Cải tiến:** Tính toán đơn giá $\text{\$/GB-VRAM-giờ}$ và tỷ số băng thông trên chi phí ($\text{TB/s / \$}$) cho 7 dòng GPU trong catalog:
  * H100: \$0.0312/GB-hr (1.34 TB/s/\$)
  * A100: \$0.0224/GB-hr (1.12 TB/s/\$)
  * A10G: \$0.0417/GB-hr (0.60 TB/s/\$)
  * MI300X: **$0.0102/GB-hr (2.72 TB/s/\$)** — tối ưu vượt trội cho memory-bound decode.
* **Kết quả:** Đề xuất chuyển 2 cụm inference A100 dư thừa VRAM sang A10G, và hạ cấp `gpu-a10g-1` (bị Util-lie) sang L4, mang lại **$1,282 / tháng** tiền tiết kiệm phần cứng bền vững.

### Extension 3: Kinh Tế Học Prompt Caching (`cache_is_worth_it`)
* **Vị trí code:** `finops/pricing.py` (`cache_is_worth_it`, `cache_break_even_reads`).
* **Công thức hòa vốn:**
  $$\text{Break-even Reads} = \frac{\text{Chi phí ghi (Write Cost)}}{\text{Đơn giá Input} \times (1 - \text{Read Discount})} = \frac{1.25 \times P_{\text{in}}}{P_{\text{in}} \times 0.90} \approx 1.39 \text{ lần đọc}$$
* **Kết quả:** Với hệ thống Chat & RAG, System Prompt có số lượt đọc trung bình là **6.5 lần** (vượt xa ngưỡng 1.39). Thuật toán xác nhận điều kiện sinh lời hợp lệ và áp dụng prompt caching.

### Extension 4: Ngân Sách & Quản Trị Truy Vấn Reasoning
* **Vị trí code:** `missions/m2_inference_levers.py`.
* **Phát hiện chấn động:**
  * Traffic Reasoning (`is_reasoning=1`) chỉ chiếm **8.4% tổng số request** (201 / 2,400).
  * Nhưng chiếm **16.5% chi phí suy luận** (\$1.40/ngày).
  * Đặc biệt tiêu thụ tới **29.79 kWh/ngày — tương đương 94.0% tổng điện năng suy luận** (do hệ số nhân năng lượng $\sim 80\times$ từ chuỗi suy luận dài).
* **Đề xuất quy tắc:** Áp dụng bộ phân loại độ phức tạp (Complexity Classifier), chỉ kích hoạt reasoning khi điểm phức tạp $\ge 0.75$ hoặc độ tự tin ban đầu $< 0.60$. Nếu cắt giảm 50% lượng reasoning không cần thiết, công ty tiết kiệm **$15/tháng** và **441.2 kWh điện/tháng**.

### Extension 5: Lập Lịch Nhận Thức Carbon (Carbon-Aware Scheduling)
* **Vị trí code:** `missions/m3_purchasing.py` (`analyze_carbon_aware_scheduling`).
* **Phân tích 5 lưới điện:**
  * `us-east-1` (Baseline): $380\text{ gCO}_2/\text{kWh}$, $\$0.12/\text{kWh}$ $\rightarrow 1,606.3\text{ kg CO}_2\text{e}$
  * `europe-north1` (Sạch nhất): $30\text{ gCO}_2/\text{kWh}$, $\$0.09/\text{kWh}$ $\rightarrow 126.8\text{ kg CO}_2\text{e}$ (**giảm 92.1% phát thải**)
  * `us-east-wa` (Rẻ nhất): $90\text{ gCO}_2/\text{kWh}$, $\$0.055/\text{kWh}$ $\rightarrow 380.4\text{ kg CO}_2\text{e}$ (**giảm 54.2% chi phí điện**, giảm 76.3% phát thải)
  * `europe-central2` (Tránh dùng): $660\text{ gCO}_2/\text{kWh}$, $\$0.18/\text{kWh}$ (đắt nhất và bẩn nhất do nhiệt điện than).
* **Trade-off Latency:** Các job training và batch evaluation là tác vụ phi đồng bộ (asynchronous). Di chuyển $4,227\text{ kWh/tháng}$ sang Na Uy (`europe-north1`) giảm trực tiếp **1.48 tấn $\text{CO}_2\text{e}$ mỗi tháng** mà không làm tăng bất kỳ millisecond trễ nào của người dùng cuối.

---

## 5. Khuyến Nghị Chiến Lược Cho Ban Giám Đốc NimbusAI

Nếu được bổ nhiệm làm FinOps Lead tại NimbusAI, 3 hành động triển khai trong 30 ngày đầu tiên:

1. **Tuần 1 — Khóa van rò rỉ tức thì (Immediate Guardrails):**
   * Kích hoạt Cronjob tự động tắt các node GPU idle qua đêm (thu hồi ngay \$600/tháng).
   * Bật Prompt Caching trên toàn bộ endpoint của RAG và Chatbot (thu hồi \$1,200/tháng).
2. **Tuần 2 — Chuyển đổi phương thức mua sắm (Smart Procurement):**
   * Tái cấu trúc pipeline training của RAG và Search sang cơ chế Spot Instance tích hợp checkpoint tự động lưu lên S3/GCS mỗi 30 phút.
   * Ký hợp đồng 1-Year Reserved cho hạ tầng suy luận cố định.
3. **Tháng 1 — Thiết lập hàng rào quản trị FOCUS & Phân quyền ngân sách (FinOps Governance):**
   * Giữ vững tỷ lệ Tag Coverage $> 90\%$ để chính thức áp dụng cơ chế **Chargeback** cho 4 nhóm (Assistant, Search, Eval, RAG).
   * Thiết lập Budget Cap cứng qua Proxy token để ngăn chặn chi tiêu ngoài kế hoạch.

---

## 6. Phụ Lục: Giải Đáp 5 Câu Hỏi Kiểm Tra Hiểu Biết Bản Chất (Oral Check)

1. **"GPU-Util 98% có nghĩa là GPU đang làm việc hiệu quả không? Tại sao?"**
   * *Trả lời:* Không. GPU-Util 98% chỉ đo tỷ lệ thời gian SM clock đang kích hoạt. Nếu mô hình bị nghẽn băng thông HBM, nghẽn I/O bus hoặc kernel launch chưa tối ưu, GPU phải chờ dữ liệu nạp vào thanh ghi (memory stall). Lúc đó MFU có thể chỉ đạt 15-20%, tức là 80% sức mạnh phần cứng bị lãng phí dù máy vẫn báo "full load".

2. **"Tại sao cần $\ge 80\%$ tag coverage mới dám chargeback?"**
   * *Trả lời:* Chargeback là hành động trừ tiền thật vào ngân sách của từng bộ phận. Nếu tag coverage $< 80\%$, hơn 20% chi phí là "vô danh" (untagged). Việc phân bổ ép buộc 20% chi phí này cho các nhóm sẽ tạo ra tranh cãi, sai lệch P&L và làm mất niềm tin vào hệ thống FinOps. Do đó, chỉ áp dụng Showback để nâng cao nhận thức, và chỉ Chargeback khi dữ liệu sạch đạt $\ge 80\%$.

3. **"Nếu công ty bạn có 70% workload interruptible, bạn sẽ tối ưu purchasing như thế nào?"**
   * *Trả lời:* Tận dụng tối đa Spot Instances kết hợp kiến trúc Fault-Tolerant Checkpointing. Đặt chu kỳ lưu checkpoint tương thích với tỷ lệ thu hồi của nhà cung cấp cloud (ví dụ: mỗi 30–60 phút). Bằng cách này, 70% workload sẽ được hưởng mức giá rẻ hơn 50–70% so với On-Demand, trong khi chi phí phụ trội do rework chỉ chiếm khoảng 3–5%.

4. **"Đo bằng $\text{\$/GPU-hr}$ vs $\text{\$/1M-token}$ — khi nào con số này cho kết quả trái ngược nhau?"**
   * *Trả lời:* Khi có sự chênh lệch lớn về hiệu quả kỹ thuật (vLLM, continuous batching, FlashAttention, cascade routing).
   * Ví dụ: Một team thuê GPU H100 giá \$2.50/giờ chạy naive serving chỉ đạt 100 tok/s $\rightarrow$ chi phí $\approx \$6.94 / \text{1M token}$. Một team khác thuê cùng GPU H100 đó nhưng tối ưu batching và prompt caching phục vụ được 2,500 tok/s $\rightarrow$ chi phí chỉ $\approx \$0.28 / \text{1M token}$. Nếu chỉ nhìn $\text{\$/GPU-hr}$ thì hai team bằng nhau, nhưng nhìn $\text{\$/1M-token}$ thì team 2 rẻ hơn **25 lần**.

5. **"Tại sao LLM decode là memory-bound còn prefill là compute-bound?"**
   * *Trả lời:* 
     * Giai đoạn **Prefill** xử lý toàn bộ prompt đầu vào cùng lúc (ma trận $\times$ ma trận, GEMM), số phép tính FLOPs tỷ lệ bậc hai với độ dài chuỗi, tạo ra Arithmetic Intensity rất cao ($\sim 455\text{ FLOP/byte}$ trên H100, vượt điểm ridge $\approx 295$), nên bị chặn bởi tốc độ tính toán (Compute-bound).
     * Giai đoạn **Decode** sinh từng token tuần tự (ma trận $\times$ vector, GEMV). Mỗi token mới sinh ra đòi hỏi phải đọc lại toàn bộ KV Cache và trọng số mô hình từ bộ nhớ HBM vào SRAM/Tensor Cores. Cường độ tính toán chỉ đạt $\approx 1 - 2\text{ FLOP/byte}$ (thấp hơn hàng trăm lần so với ridge point), khiến GPU liên tục phải chờ băng thông bộ nhớ (Memory-bound).
