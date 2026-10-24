# Firebreak Hardened R1 — ghi chú bản sửa đã kiểm tra

## Bảo toàn bản gốc
- Không ghi đè `firebreak_selective_merge_download(1).py`.
- Bản giao mới giữ kiến trúc Streamlit, mô hình CA, cấu trúc min-cut và các tham số/giao diện không liên quan. Các thay đổi chỉ tập trung vào lớp dữ liệu, tính nhất quán của vật cản, điều kiện thời gian, trạng thái khi đường băng hoàn tất muộn và các mô tả chưa có bằng chứng.
- Có bộ test hồi quy nhỏ và validator độ phủ dữ liệu. Diff đầy đủ của vòng sửa trước nằm ở file `firebreak_hardened_r1.diff` trong workspace gốc, nhưng không cần để chạy gói phát hành.

## Thay đổi cụ thể
1. Ưu tiên GeoTIFF DEM cục bộ và ESA WorldCover 2021 v200 `Map`; chỉ chấp nhận tile WorldCover có tên file thể hiện `2021`, `v200`, `Map`, tránh đọc nhầm `InputQuality`.
2. Thêm mặt nạ nước từ polygon OSM/đường bờ có nhận diện được và WorldCover class 80; nước, đường, khu dân cư và mọi ô nhiên liệu bằng 0 không được coi là ô cháy được trong CA/trường thời gian đến liên tục.
3. Loại cạnh Dijkstra liên tục đi vào ô không cháy được; chặn cắt góc chéo giữa các vật cản trong CA, Dijkstra và kiểm định topo theo cùng giả định lân cận.
4. Không tự dời tâm cháy ra một vị trí bất kỳ khi không thể khởi tạo tại vị trí/sai số đã khai báo; báo lỗi rõ ràng.
5. Thời gian tiếp cận dùng P90 trên tất cả pixel thi công; nếu bất kỳ pixel nào không tiếp cận được thì ứng viên không khả thi. Thời gian lửa chạm dùng thời điểm sớm nhất được mô hình ước tính trừ một timestep dự phòng.
6. Khi đường băng được kích hoạt, không ghi đè pixel đã cháy/đang cháy thành `STATE_FIREBREAK`; lửa trên tuyến vẫn được giữ trong lịch sử và làm phương án thất bại ở kiểm định động.
7. Xóa/cập nhật các tuyên bố không được code hỗ trợ; sửa nhãn hướng gió và nhãn metric tiếp cận P90; bỏ tuyên bố hiệu năng 8×/20× chưa có benchmark.
8. Cache Streamlit vẫn giúp tái sử dụng dữ liệu đã tải, nhưng ứng dụng **không được tuyên bố hoàn toàn offline**: cache miss vẫn cần Overpass/OpenTopoData/Open-Meteo nếu không có đủ dữ liệu cục bộ.

## Kiểm tra đã chạy trong môi trường này
- `python -m py_compile firebreak_hardened_r1.py validate_firebreak_data.py`: đạt.
- `python -m pytest -q tests/test_core.py`: **9 passed**. Test gọi các hàm thuần được trích xuất từ module; không khởi động Streamlit hoặc gọi API trực tiếp.
- `python validate_firebreak_data.py`: chạy được; hiện không có DEM/WorldCover GeoTIFF trong gói nên tất cả preset báo `MISSING` như dự kiến.
- Môi trường đóng gói hiện không có module `streamlit`, vì vậy chưa thể chạy smoke test toàn ứng dụng trong container này. Máy người dùng cần cài `requirements.txt` rồi chạy app.

## Chưa được xác thực — cần nói rõ trong báo cáo
- Chưa có DEM/WorldCover thực trong gói; phải tải tile và chạy validator/kiểm tra coverage.
- Chưa kiểm tra tích hợp với tất cả Overpass mirrors, OpenTopoData và Open-Meteo tại máy người dùng; chưa xác nhận cả năm vùng mẫu.
- OSM coastline/water có thể thiếu hoặc không đầy đủ; multipolygon relations không được xử lý tổng quát. Kiểm tra thủ công lớp nước, đặc biệt vùng biển Sơn Trà và các hồ.
- WorldCover là phân loại lớp phủ, không phải dữ liệu nhiên liệu/sinh khối/độ ẩm đã hiệu chuẩn; các hệ số nhiên liệu vẫn là heuristic.
- DEM SRTM 30 m có thể cải thiện đáng kể so với lưới API 20×20, nhưng không tự xác thực mô hình cháy, Tobler hay năng suất thi công.
- Các hệ số cháy và hiệu quả đường băng chưa được xác thực ngoài thực địa. Chỉ báo cáo là kết quả trong mô hình, không phải hướng dẫn tác chiến hay dự báo vận hành.
