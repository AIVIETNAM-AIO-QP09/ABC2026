# ABC 2026 — Nhận diện vị trí trong nhà từ BLE

![Luồng xử lý từ dữ liệu BLE gốc đến dự đoán vị trí](docs/assets/pipeline.svg)

[English](README.md) · [Chuẩn dữ liệu](docs/data-contract.md) · [Kiến trúc](docs/architecture.md) · [Bằng chứng nghiên cứu](docs/research.md) · [Ghi chú chuyển đổi](docs/migration.md)

Repo chuyển các notebook nghiên cứu của ABC 2026 thành một package Python để chuẩn bị dữ liệu Bluetooth Low Energy (BLE) tầng 5, huấn luyện mô hình **Confidence-Guided Cycle Retraining (CGCR)** và dự đoán phòng từ cường độ tín hiệu RSSI. Quy trình huấn luyện cuối dựa trên phần mã hiện còn của nghiên cứu *From Noisy Beacons to Precise Location: A Confidence-Guided Cycle Retraining Strategy*.

> **Phạm vi kiểm chứng:** điểm Leave-One-Day-Out (LODO) bên dưới là kết quả **paper báo cáo**. Repo hiện cung cấp pipeline huấn luyện mô hình cuối, chưa tái lập phép đánh giá LODO của paper. Notebook tiền xử lý cũ bị thất lạc; bước này được dựng lại từ dữ liệu gốc và hai tutorial của ban tổ chức, với các giả định được công khai.

## Mục lục

- [Bài toán và dữ liệu](#bài-toán-và-dữ-liệu)
- [Phương pháp](#phương-pháp)
- [Chạy nhanh](#chạy-nhanh)
- [Đầu vào và đầu ra](#đầu-vào-và-đầu-ra)
- [Kết quả nghiên cứu và khả năng tái lập](#kết-quả-nghiên-cứu-và-khả-năng-tái-lập)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Phát triển](#phát-triển)

## Bài toán và dữ liệu

Mục tiêu là nhận diện phòng hoặc khu vực chung trên tầng 5 từ RSSI của các beacon BLE. Thiết bị của **user 90** thu tín hiệu từ **25 beacon đã ánh xạ**; bản ghi vị trí của **user 97** do ban tổ chức cung cấp làm nhãn huấn luyện. Dữ liệu được thu từ ngày 10 đến 13/04/2023. RSSI biến động theo khoảng cách, vật cản, hướng thiết bị và ngày thu thập, nên một phép đo riêng lẻ thường nhiễu.

![Sơ đồ tầng 5 của ban tổ chức với vị trí beacon được đánh số](docs/floor-map-source.png)

ZIP gốc chứa các tệp sự kiện BLE không có header và nhãn theo khoảng thời gian. `config/beacon_map.json` lưu đúng thứ tự MAC → beacon trong tutorial. Lệnh `prepare` đọc trực tiếp ZIP, tạo mỗi dòng ứng với một giây và các cột `rssi_1` đến `rssi_25`; giá trị 0 nghĩa là không ghi nhận beacon đó. Bản dựng lại lấy **RSSI mạnh nhất của từng beacon trong giây** và chỉ gán nhãn khi cả giây nằm trọn trong một khoảng nhãn không xung đột. Các giây còn lại được để không nhãn.

Với ZIP được cung cấp, bộ lọc chọn **1.673.395 sự kiện BLE**, trùng mốc kiểm tra trong tutorial; kết quả là **23.202 dòng có nhãn** và **13.818 dòng không nhãn**. Các số này mô tả bản tiền xử lý hiện tại, không chứng minh notebook cũ dùng cùng cách gom giây và ghép nhãn. Xem [chuẩn dữ liệu](docs/data-contract.md) để biết schema và nguồn gốc từng quy tắc.

## Phương pháp

![Hình từ paper: dữ liệu tăng cường, đặc trưng, pseudo-label và huấn luyện theo chu kỳ](docs/assets/paper-framework.png)

1. **Chuẩn bị dữ liệu:** lọc ZIP theo user, thời gian và danh sách beacon của ban tổ chức; chuyển sự kiện và nhãn thành bảng wide theo giây.
2. **Tăng cường lớp ít mẫu:** tạo các đoạn tín hiệu tổng hợp từ chuỗi của những ngày huấn luyện. Bộ sinh giữ các quy tắc proxy theo fold cho phòng 503 và 510. Dữ liệu có seed nhưng chỉ là bản dựng lại quy trình, không cam kết trùng từng dòng với CSV lịch sử.
3. **Trích xuất đặc trưng:** tạo cửa sổ năm giây không chồng lấp bằng cùng một đường xử lý cho dữ liệu có nhãn và không nhãn. Đặc trưng gồm RSSI trung bình, cosine similarity với chữ ký tín hiệu từng phòng, hiệu giữa các beacon, tọa độ không gian có trọng số tín hiệu, độ lệch bắc–nam và ngữ cảnh cuộn. Loại cửa sổ có nhãn qua ranh giới phòng và `hallway`.
4. **Huấn luyện C0–C3:** C0 dùng các cửa sổ có nhãn thật và dữ liệu tổng hợp. Sau mỗi chu kỳ, mô hình gán nhãn cho dữ liệu chưa có nhãn; chỉ những dự đoán vượt ngưỡng của **chu kỳ tiếp theo** mới được dùng với trọng số tương ứng. Tập pseudo-label được tạo lại sau mỗi lần fit.
5. **Suy luận:** lưu mô hình cùng thứ tự đặc trưng, bộ mã hóa nhãn, chữ ký phòng và cấu hình. Khi dự đoán, làm mượt lớp của các cửa sổ bằng bỏ phiếu đa số trên năm cửa sổ, rồi ánh xạ lại từng dòng test theo đúng thứ tự gốc.

![Hình từ paper: ba giai đoạn thích nghi, mở rộng và ổn định](docs/assets/paper-cycle-retraining.png)

| Chu kỳ huấn luyện | Ngưỡng chọn pseudo-label đầu vào | Trọng số pseudo-label |
| --- | ---: | ---: |
| C0 | — | — |
| C1 | 0,85 | 0,30 |
| C2 | 0,90 | 0,60 |
| C3 | 0,95 | 0,45 |

Cấu hình này lấy từ Bảng I của paper và được cài đặt trong [`src/cgcr/config.py`](src/cgcr/config.py). Mô hình cũng cân bằng trọng số lớp và tăng trọng số cho một số lớp yếu. Xem [kiến trúc](docs/architecture.md) để hiểu vai trò từng module.

## Chạy nhanh

Cần **Python 3.10+** và [`uv`](https://docs.astral.sh/uv/). Chạy PowerShell tại thư mục gốc repo. Có thể đặt ZIP ở bất cứ đâu; dữ liệu và model sinh ra bên dưới đã được loại khỏi Git.

```powershell
uv venv
uv pip install -e .

$archive = 'C:\path\to\ABC2026.zip'
uv run cgcr inspect-raw --archive $archive --report outputs/mac_inventory.csv
uv run cgcr prepare --archive $archive --beacon-map config/beacon_map.json --output-dir data/prepared
uv run cgcr augment --wide data/prepared/labeled_wide.csv --output data/prepared/synthetic.csv --repeats 6 --seed 42
uv run cgcr train --labeled data/prepared/labeled_wide.csv --synthetic data/prepared/synthetic.csv --unlabeled data/prepared/unlabeled_wide.csv --model models/cgcr.joblib
```

`inspect-raw` không bắt buộc, chỉ xuất báo cáo MAC để kiểm tra. `prepare` đọc ZIP trực tiếp, không cần giải nén. Bước huấn luyện thường lâu hơn tiền xử lý vì fit bốn mô hình XGBoost.

Khi có CSV test wide riêng:

```powershell
uv run cgcr predict --model models/cgcr.joblib --test data/test_file.csv --output-dir outputs
```

ZIP của ban tổ chức **không chứa** `test_file.csv` trong notebook cuối. Notebook lịch sử dùng `fakedata_labeled_yes_2.csv` làm dữ liệu tổng hợp; nếu còn file đó, có thể đưa trực tiếp vào `--synthetic` để dùng đúng artifact trung gian. Lệnh `augment` dựng lại quy tắc sinh dữ liệu, không bảo đảm tạo CSV giống hệt.

## Đầu vào và đầu ra

| Bước | Đầu vào chính | Đầu ra |
| --- | --- | --- |
| `inspect-raw` | ZIP gốc | `outputs/mac_inventory.csv` |
| `prepare` | ZIP + `config/beacon_map.json` | `data/prepared/labeled_wide.csv`, `unlabeled_wide.csv`, `preprocessing_summary.json` |
| `augment` | CSV wide có nhãn | CSV tổng hợp có `location`, các cột RSSI và metadata |
| `train` | CSV wide có nhãn, tổng hợp và không nhãn | Bundle `models/cgcr.joblib` |
| `predict` | Bundle + CSV wide test riêng | `outputs/predictions_windows.csv` và `predictions_rows.csv` |

CSV wide cần `timestamp` có thể phân tích và các cột `rssi_<id>`; bảng có nhãn cần `room` hoặc `location`. Loader dữ liệu tổng hợp cũng nhận định dạng lịch sử. File dự đoán theo cửa sổ có xác suất từng phòng và xác suất lớp cao nhất **trước khi làm mượt**; nhãn phòng cuối cùng có thể đổi sau khi bỏ phiếu đa số. File theo dòng giữ dữ liệu test gốc và thêm `predicted_room`. Xem [chuẩn dữ liệu](docs/data-contract.md) về beacon bị thiếu và tín hiệu không hoạt động.

## Kết quả nghiên cứu và khả năng tái lập

Paper báo cáo các **giá trị trung bình LODO theo ngày được giữ lại** sau đây (Bảng II). Đây **không phải kết quả chạy từ package đã tái cấu trúc**.

| Chu kỳ | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| C0 | 0,6986 | 0,5742 | 0,7071 |
| C1 | 0,7036 | 0,5848 | 0,7101 |
| C2 | 0,7079 | 0,6131 | 0,7150 |
| **C3** | **0,7091** | **0,6220** | **0,7166** |

![Macro F1 của C0–C3 theo kết quả LODO trong paper](docs/assets/paper-macro-f1.svg)

Bảng ablation của paper còn báo cáo Macro F1 **0,4978** khi chỉ dùng đặc trưng và **0,6220** cho toàn bộ framework. Đây là kết quả của bản PDF được cung cấp; xem [bằng chứng nghiên cứu](docs/research.md) và paper gốc để biết độ biến thiên theo fold và thiết kế đánh giá.

| Đã kiểm tra trong repo | Còn cần làm để tái lập đầy đủ |
| --- | --- |
| Ánh xạ MAC đúng tutorial; bộ lọc dữ liệu gốc khớp mốc số sự kiện BLE. | Chưa biết chính xác cách gom dữ liệu wide và chia train/test của notebook tiền xử lý đã mất. |
| Đã chạy tiền xử lý và sinh dữ liệu tổng hợp có seed trên ZIP được cung cấp; các test contract cơ bản đều đạt. | Không tái tạo CSV tổng hợp lịch sử giống từng byte. |
| Code huấn luyện cuối và suy luận đã được tổ chức trong package. | Chưa chạy lại huấn luyện XGBoost bốn chu kỳ và bảng LODO trong paper. |

Đánh giá LODO nghiêm ngặt cần tách dữ liệu theo ngày và chỉ chuẩn bị dữ liệu trong phần train của từng fold. Quy trình tăng cường được dựng lại có bước chọn seed dựa vào thông tin fold; **không được để nhãn validation đi vào pipeline đánh giá mù**. Lệnh `train` hiện huấn luyện mô hình cuối trên dữ liệu có nhãn được cung cấp và **không in điểm accuracy**. Xem thêm [ghi chú chuyển đổi](docs/migration.md).

## Cấu trúc repo

```text
config/beacon_map.json       Ánh xạ MAC → ID beacon từ tutorial
src/cgcr/raw.py              Kiểm tra ZIP và tạo bảng wide theo giây
src/cgcr/augmentation.py     Sinh đoạn tổng hợp cho lớp ít mẫu
src/cgcr/features.py         Đặc trưng cửa sổ, RSSI, không gian và ngữ cảnh
src/cgcr/pipeline.py         Huấn luyện C0–C3 và dự đoán
src/cgcr/cli.py              Giao diện dòng lệnh cgcr
tests/                       Kiểm tra contract dữ liệu và pipeline
docs/                        Kiến trúc, dữ liệu, nghiên cứu, ghi chú chuyển đổi
notebooks/legacy/            Notebook nghiên cứu gốc, giữ nguyên
notebooks/organizer/         Tutorial ban tổ chức, giữ nguyên
```

Notebook được giữ để truy xuất nguồn gốc; code bảo trì nằm trong `src/cgcr/`. Dữ liệu, model và file dự đoán tạo ra được loại khỏi Git bằng `.gitignore`.

## Phát triển

```powershell
uv pip install -e '.[dev]'
uv run pytest
uv run ruff check src tests
```

GitHub Actions chạy các test cơ bản trên Python 3.11 và 3.12. Test kiểm tra contract và các phép biến đổi quan trọng; chúng chưa thay thế phép tái lập LODO đầu cuối.
