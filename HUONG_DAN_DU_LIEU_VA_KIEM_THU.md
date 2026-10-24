# Hướng dẫn dữ liệu và kiểm thử — Firebreak Hardened R1

## 1. Cấu trúc thư mục

Giữ file `firebreak_hardened_r1.py` và tạo các thư mục bên cạnh nó:

```text
project/
├── firebreak_hardened_r1.py
├── requirements.txt
├── data/
│   ├── dem/
│   │   ├── dem_sapa.tif
│   │   ├── dem_dalat.tif
│   │   └── ...
│   └── worldcover/
│       ├── ESA_WorldCover_10m_2021_v200_NxxExxx_Map.tif
│       ├── ESA_WorldCover_10m_2021_v200_NxxExxx_Map.tif
│       └── ...
└── tests/
    └── test_core.py
```

Có thể đặt nhiều tile cho các vùng mẫu khác nhau trong mỗi thư mục. Bản R1 đọc các tile giao với ROI và tái chiếu trực tiếp lên lưới tính toán (không tạo mosaic khổng lồ ở độ phân giải nguồn). Nếu các tile chồng lấn, ô đã được một tile hợp lệ cung cấp sẽ không bị ghi đè bởi tile tiếp theo; hãy dùng bộ tile cùng sản phẩm/phiên bản và giữ nguyên tên file. Với WorldCover, code chỉ đọc tên tile gốc có cả `2021`, `v200` và `Map`, tránh đọc nhầm `InputQuality`.

## 1A. Khung tọa độ cần tải dữ liệu cho 5 vùng mẫu

Các khung sau là ROI code thực sự dùng (tâm ±0,075°), không phải ranh giới toàn bộ vườn quốc gia/khu rừng:

| Preset | Kinh độ Tây–Đông | Vĩ độ Nam–Bắc |
|---|---:|---:|
| Hoàng Liên Sơn – Sa Pa | 103.7500–103.9000 | 22.3000–22.4500 |
| Rừng Thông Đà Lạt | 108.3833–108.5333 | 11.8654–12.0154 |
| Ba Vì | 105.2850–105.4350 | 20.9850–21.1350 |
| Hồng Lĩnh | 105.6450–105.7950 | 18.4450–18.5950 |
| Sơn Trà | 108.1900–108.3400 | 16.0400–16.1900 |

**Tải WorldCover nhanh nhất:** mở trang Data Access chính thức, chọn WorldCover 2021 v200, mở WorldCover viewer, đăng ký/đăng nhập Terrascope, mở tab `Download`, vẽ polygon/rectangle phủ từng ROI, tải tất cả tile của lớp `Map` và đặt các `.tif` vào `data/worldcover/`. WorldCover được phân phối theo tile 3°×3°; tải tile bao phủ ROI thay vì đồng bộ toàn bộ bucket (khoảng 117 GB). ROI Ba Vì cắt qua vĩ độ 21° và ROI Đà Lạt cắt qua vĩ độ 12°, nên hãy để viewer chọn đủ tile hai phía mỗi ranh đó. (Nguồn: ESA WorldCover, https://esa-worldcover.org/en/data-access)

**Tải DEM:** mở EarthExplorer, đăng ký tài khoản, vào `Search Criteria` nhập khung ROI (hoặc nhập tâm rồi chỉnh polygon), vào `Data Sets` → `Digital Elevation` → `SRTM` → chọn `SRTM 1 Arc-Second Global` và định dạng `GeoTIFF`. Tải các tile giao với ROI, giải nén rồi đặt tất cả file `.tif`/`.tiff` vào `data/dem/`. USGS xác nhận SRTM 1 arc-second tương đương khoảng 30 m và có thể tải dạng GeoTIFF qua EarthExplorer; độ phân giải danh nghĩa không có nghĩa là mọi chi tiết địa hình đều chính xác 30 m. (Nguồn: USGS, https://www.usgs.gov/faqs/where-can-i-get-global-elevation-data)

Sau khi chép file, chạy validator ở mục 5. Mục tiêu: `OK (tile bounds cover ROI)` cho cả DEM và WorldCover ở từng preset; nếu báo `MISSING` hoặc `PARTIAL`, chưa nên chốt số liệu cho vùng đó.

## 2. Lớp phủ đất và nước — ưu tiên lấy trước

**Nguồn đề xuất: ESA WorldCover 2021 v200, bản đồ lớp phủ 10 m.** Trang truy cập dữ liệu chính thức: <https://esa-worldcover.org/en/data-access>. Chọn tải lớp *Map* dạng GeoTIFF cho từng vùng mẫu; không tải lớp `InputQuality` thay cho bản đồ lớp phủ và giữ nguyên tên tile gốc có `2021_v200` trong tên. ESA phân phối bản đồ theo các tile GeoTIFF tối ưu cloud; trang dữ liệu cũng cung cấp hướng dẫn tải bằng trình tải của ESA, Zenodo và AWS. Xem tài liệu chính thức để biết giấy phép và trích dẫn. (Nguồn: https://esa-worldcover.org/en/data-access)

Trong code hiện tại, cách phân loại ban đầu là:

| Mã WorldCover | Cách code xử lý | Ghi chú |
|---:|---|---|
| 10 — Tree cover | Có thể cháy | Không phải bản đồ sinh khối/độ ẩm nhiên liệu |
| 20 — Shrubland | Có thể cháy | Là proxy thực bì |
| 30 — Grassland | Có thể cháy | Là proxy thực bì |
| 40 — Cropland | Có thể cháy | Phụ thuộc mùa vụ và loại cây; cần hiệu chỉnh cho nghiên cứu rừng |
| 80 — Permanent water bodies | Không cháy; mặt nước | Gồm nước được phân loại trong lớp raster |
| 95 — Mangroves | Có thể cháy | Vẫn là giả định đơn giản hóa |
| Các lớp còn lại | Không tự động coi là nhiên liệu | Wetland/bare land có thể cần quy tắc địa phương riêng; ô này không bị tính là pixel cần phát quang mới |

**Lưu ý quan trọng:** đây là mặt nạ khả năng cháy ban đầu, không phải mô hình nhiên liệu đã hiệu chuẩn. WorldCover 2021 v200 được ESA báo cáo độ chính xác tổng thể 76,7%, vì vậy vẫn có sai số phân loại; hãy kiểm tra thủ công riêng hồ, bờ biển và ranh rừng của ROI. OSM water polygons được dùng để che thêm các thủy hệ được gắn thẻ rõ, nhưng quan hệ/multipolygon OSM không được xử lý đầy đủ trong parser hiện tại. Không được công bố diện tích cháy là diện tích rừng thực tế nếu chưa kiểm tra lớp thảm phủ. (Nguồn: ESA WorldCover, https://esa-worldcover.org/en/data-access)

## 3. DEM — thay thế mẫu độ cao thưa từ API

**Nguồn khả thi: USGS EarthExplorer — SRTM 1 Arc-Second Global (khoảng 30 m).** USGS cung cấp sản phẩm SRTM 1 arc-second ở dạng GeoTIFF; việc tải dữ liệu yêu cầu đăng ký tài khoản. Bắt đầu từ trang hướng dẫn dữ liệu độ cao toàn cầu của USGS: <https://www.usgs.gov/faqs/where-can-i-get-global-elevation-data>, sau đó vào EarthExplorer và chọn dataset SRTM 1 Arc-Second Global cho vùng cần lấy. (Nguồn: USGS, https://www.usgs.gov/faqs/where-can-i-get-global-elevation-data)

Đặt tất cả tile GeoTIFF đã tải vào `data/dem/`. Với SRTM, khung mô phỏng có thể cắt qua ranh tile độ 1°: ROI Ba Vì cắt vĩ độ 21° và ROI Đà Lạt cắt vĩ độ 12°. Tải đủ các tile giao với từng ROI, kể cả hai phía ranh này. Bản R1 ghép tile lên lưới đích. Nếu không có tile DEM địa phương nào, code quay lại OpenTopoData và cảnh báo rằng DEM API được lấy mẫu 20×20 rồi nội suy — không phải dữ liệu địa hình thực 30 m. Nếu đã đặt tile DEM cục bộ nhưng còn ô thiếu trên đất liền, code dừng với thông báo cần bổ sung tile thay vì âm thầm trộn nguồn độ cao hoặc tự tạo địa hình.

Copernicus DEM GLO-30 cũng là nguồn DSM 30 m toàn cầu, nhưng quyền tải hiện phụ thuộc nhóm người dùng đủ điều kiện của Copernicus Data Space; kiểm tra điều kiện truy cập hiện hành trước khi chọn nó. (Nguồn: Copernicus Data Space, https://dataspace.copernicus.eu/)

## 4. Dữ liệu để kiểm chứng mô hình cháy

Không nên hiệu chuẩn mô hình chỉ bằng dữ liệu địa hình/thảm phủ. Chọn một vụ cháy đã xảy ra, xác định ngày và phạm vi độc lập, rồi thu thập:

1. **Hotspot để tìm sự kiện:** NASA FIRMS cung cấp điểm phát hiện cháy MODIS/VIIRS và kho dữ liệu lịch sử; VIIRS có sản phẩm 375 m. Điểm hotspot không phải ranh giới đám cháy và không có nghĩa toàn bộ pixel vệ tinh đã cháy. Dùng để xác định thời gian/khu vực sự kiện, không dùng làm ground truth chính xác cho đường biên. <https://firms.modaps.eosdis.nasa.gov/download/> (Nguồn: NASA FIRMS, https://firms.modaps.eosdis.nasa.gov/download/)
2. **Ranh giới vùng cháy:** ưu tiên ranh giới do cơ quan chuyên môn công bố hoặc lập bản đồ vùng cháy từ ảnh vệ tinh trước/sau sự kiện, với phương pháp và ngày ảnh được ghi rõ.
3. **Burnt-area để đối chiếu quy mô lớn:** ưu tiên Copernicus Land Monitoring Service Burnt Area v4: sản phẩm 300 m hằng ngày từ năm 2025 đến hiện tại và sản phẩm 300 m theo tháng từ năm 2018 đến hiện tại. Đây là dữ liệu đốt cháy ở quy mô vệ tinh; phù hợp đối chiếu quy mô/tính nhất quán thời gian, không xác nhận một đường băng rộng vài pixel. <https://land.copernicus.eu/en/products/vegetation/burnt-area-v4-daily-300m> (Nguồn: Copernicus Land Monitoring Service, https://land.copernicus.eu/en/products/vegetation/burnt-area-v4-daily-300m)
4. **Thời tiết:** endpoint trong ứng dụng lấy thời tiết hiện tại. Với sự kiện lịch sử, phải dùng dữ liệu khí tượng lịch sử phù hợp ngày xảy ra cháy; không dùng thời tiết hôm nay để xác minh một đám cháy trong quá khứ.

Ghi lại thời điểm tải, nguồn, phiên bản/lớp dữ liệu, license và checksum. Giữ một tập sự kiện để phát triển và một tập độc lập để đánh giá; không điều chỉnh hệ số trên cùng một sự kiện rồi gọi chính sự kiện đó là xác thực độc lập.

## 5. Cài đặt và chạy kiểm thử

Trong PowerShell, tại thư mục dự án:

```powershell
python -m pip install -r requirements.txt
python -m py_compile firebreak_hardened_r1.py
python -m pytest -q tests
python -m streamlit run firebreak_hardened_r1.py
```

Bộ test hiện tại có 9 kiểm thử tự động cho các hàm thuần được trích xuất từ ứng dụng: đọc đúng WorldCover Map, loại đường truyền Dijkstra qua ô không cháy, ngăn CA cắt góc qua vật cản, tránh xóa trạng thái cháy khi đường băng kích hoạt muộn, và kiểm tra điều kiện tiếp cận P90. Chúng **không** thay thế cho chạy thử toàn ứng dụng với API, kiểm tra trực quan năm vùng mẫu, benchmark 540×540, phân tích độ nhạy nhiều seed hoặc xác thực ngoài thực địa.

## 5A. Hành vi cache và kiểm thử hồi quy mới

- Ứng dụng lập khóa cache từ tên file, kích thước và thời điểm sửa đổi của các GeoTIFF trong `data/dem/` và `data/worldcover/`. Khi thêm, xóa hoặc cập nhật tile, khóa cache thay đổi để môi trường địa lý được nạp lại; không cần đưa raster lớn vào Git.
- Các waterway OSM có tag `river`, `canal`, `stream` hoặc `riverbank` được raster hóa thành hành lang nước không cháy. Đối tượng dạng đường vẫn là proxy hẹp, không thay thế polygon mặt nước được lập bản đồ.
- Bộ kiểm thử bao gồm các trường hợp cho từng tag waterway, kiểm tra ngăn đi chéo qua góc bị chặn trong kiểm định tô-pô, và kiểm tra invalidation khóa cache khi tập tile thay đổi.
- Khi ước lượng chân trời lập kế hoạch từ mặt trận ban đầu, ô không cháy được (bao gồm nước và các lớp WorldCover có fuel bằng 0) không được tính là pixel phát quang.

## 6. Giới hạn còn tồn tại cần ghi trong báo cáo

- Không có dữ liệu WorldCover thì fallback OSM thực vật không đầy đủ; vùng không được OSM gắn thẻ sẽ không được tự động coi là rừng. Đây là chế độ an toàn hơn việc giả định toàn bộ đất liền là rừng, nhưng có thể bỏ sót vùng nhiên liệu.
- Mặt nạ WorldCover 10 m được resample lên lưới tính toán; nó không cung cấp chi tiết nhiên liệu/sinh khối cấp pixel 30 m hoặc độ ẩm thực địa.
- DEM cục bộ cải thiện dữ liệu độ cao đầu vào, nhưng không tự chứng minh công thức lan lửa, tốc độ hành quân hay năng suất thi công là đúng.
- Các nguồn Overpass/Open-Meteo vẫn cần kết nối Internet. Việc cài DEM/WorldCover cục bộ không biến toàn bộ ứng dụng thành ứng dụng offline.
- Trạng thái “kiểm định động” chỉ đề cập tới tiêu chí nội bộ của mô phỏng; không phải chứng nhận an toàn chữa cháy ngoài thực địa.
