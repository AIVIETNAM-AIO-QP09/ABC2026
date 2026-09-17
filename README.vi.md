# ABC 2026: Nhận diện vị trí trong nhà bằng BLE

[English](README.md) · [Sơ đồ hệ thống](docs/architecture.md) · [Chuẩn dữ liệu](docs/data-contract.md)

Repo này chuyển các notebook nghiên cứu cũ thành package Python có thể chạy theo từng bước. Notebook gốc được giữ nguyên trong `notebooks/legacy/`; hai tutorial của ban tổ chức nằm trong `notebooks/organizer/`. Mã chạy chính nằm ở `src/cgcr/`.

## Chạy từ dữ liệu gốc

Cần Python 3.10+ và `uv`. Chạy tại thư mục gốc của repo:

```powershell
uv venv
uv pip install -e .
$archive = 'C:\path\to\ABC2026.zip'
uv run cgcr prepare --archive $archive --beacon-map config/beacon_map.json --output-dir data/prepared
uv run cgcr augment --wide data/prepared/labeled_wide.csv --output data/prepared/synthetic.csv
uv run cgcr train --labeled data/prepared/labeled_wide.csv --synthetic data/prepared/synthetic.csv --unlabeled data/prepared/unlabeled_wide.csv --model models/cgcr.joblib
```

Khi có file test riêng:

```powershell
uv run cgcr predict --model models/cgcr.joblib --test data/test_file.csv --output-dir outputs
```

ZIP được đọc trực tiếp, không cần giải nén. `config/beacon_map.json` chứa bảng MAC → beacon 1–25 đúng thứ tự trong tutorial của ban tổ chức. File test dùng trong notebook cuối không có trong ZIP.

## Đầu ra và giới hạn

`prepare` tạo `labeled_wide.csv`, `unlabeled_wide.csv` và `preprocessing_summary.json`. `augment` tạo dữ liệu bổ sung cho lớp hiếm. `train` lưu model cùng thứ tự đặc trưng, nhãn và room signatures. `predict` xuất dự đoán theo cửa sổ và theo từng dòng test. Các file dữ liệu, model và đầu ra không được đưa vào Git.

Notebook tiền xử lý cũ vẫn thất lạc. Bước ghép dữ liệu gốc hiện tại là bản dựng lại có quy tắc rõ ràng: RSSI mạnh nhất trong mỗi giây, chỉ gán nhãn cho giây nằm trọn trong một khoảng nhãn không xung đột. Vì vậy không nên coi các số liệu của paper là kết quả đã được tái lập từ repo này. Chi tiết nằm trong [chuẩn dữ liệu](docs/data-contract.md) và [ghi chú chuyển đổi](docs/migration.md).
