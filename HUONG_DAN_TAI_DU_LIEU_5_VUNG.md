# Tải DEM + ESA WorldCover cho 5 vùng Firebreak

## Tải nhanh trên Windows

1. Đặt `download_firebreak_data.py` và `TAI_DU_LIEU_FIREBREAK.bat` vào **cùng thư mục** với file dashboard `firebreak_hardened_r1.py`.
2. Nhấp đúp `TAI_DU_LIEU_FIREBREAK.bat`. Nếu Windows không có lệnh `py`, mở PowerShell ở thư mục dự án và chạy:
   ```powershell
   python -m pip install requests
   python download_firebreak_data.py
   ```
3. Script chỉ lấy những tile giao với 5 vùng preset và lưu thẳng vào `data\worldcover\` và `data\dem\` — đúng các thư mục dashboard hiện đọc.
4. Nếu tải bị ngắt, chạy lại lệnh. Các TIFF hoàn chỉnh đã có sẽ được bỏ qua; file tải dở không được tính là hợp lệ.
5. Khi xong, chạy kiểm tra dữ liệu có trong project nếu có `validate_firebreak_data.py`:
   ```powershell
   python validate_firebreak_data.py
   ```
   Sau đó chạy dashboard:
   ```powershell
   python -m streamlit run firebreak_hardened_r1.py
   ```

## Khu vực và tile dự kiến

Phạm vi mỗi vùng là 0.15° × 0.15° đúng theo `LAT_SPAN_DEG` và `LON_SPAN_DEG` trong code. Script đọc tile index của ESA để lựa chọn WorldCover; nếu index không truy cập được, nó suy ra tile theo lưới 3° × 3° đã công bố.

Các tile WorldCover dự kiến (tile ID được ghi theo góc tây-nam):
- Hoàng Liên Sơn – Sa Pa: `N21E102`
- Đà Lạt: `N09E108`, `N12E108` (AOI cắt qua vĩ độ 12°)
- Ba Vì: `N18E105`, `N21E105` (AOI cắt qua vĩ độ 21°)
- Hồng Lĩnh: `N18E105` (dùng chung tile với phần tile của Ba Vì)
- Sơn Trà: `N15E108`

Tổng dự kiến: 6 tile WorldCover duy nhất. Script không chỉ tin danh sách trên: nó sẽ đối chiếu index tile chính thức nếu truy cập được.

Các ô Copernicus DEM dự kiến là 1° × 1°; những AOI cắt qua ranh vĩ độ 12° và 21° sẽ lấy cả hai phía. Script thử GLO-30 Public trước, rồi mới thử GLO-90 nếu tile GLO-30 không công khai.

## Ý nghĩa và giới hạn dữ liệu

- **ESA WorldCover 2021 v200 Map**: lớp phủ đất danh nghĩa 10 m, 11 lớp. Đây không phải số đo sinh khối/nhiên liệu cháy. Các hệ số nhiên liệu trong ứng dụng vẫn là giá trị heuristic, chưa được hiệu chuẩn theo địa phương.
- **Copernicus DEM GLO-30**: mô hình bề mặt số (DSM), có thể bao gồm cây cối và công trình, không phải DEM mặt đất trần. Nếu cần địa hình trần chính xác cho địa hình rừng, phải đánh giá thêm nguồn DEM/DTM phù hợp.
- **Fallback GLO-90**: độ phân giải thấp hơn; cần ghi nhận tile nào dùng fallback khi báo cáo.
- Script kiểm tra HTTP, kích thước sơ bộ và chữ ký TIFF. Đây chưa phải kiểm định toàn bộ dữ liệu, CRS, NoData hoặc độ phủ raster. Dashboard/validator vẫn cần chạy sau khi tải.
- OSM vẫn cung cấp đường/khu dân cư/nước theo thời điểm truy vấn; các lớp OSM có thể không đầy đủ. WorldCover lớp 80 là nước mặt được phân loại, không thay thế hoàn toàn bản đồ thủy văn độ chi tiết cao.
- Mô phỏng lưới 540×540 có pixel đầu ra danh nghĩa 30 m, nhưng dữ liệu đầu vào được resample/reproject về lưới đó. Không được gọi mọi đầu vào là độ phân giải thực 30 m nếu có GLO-90 hoặc nguồn khác.

## Nguồn và ghi công

- ESA WorldCover dataset: Zanaga et al. (2022), *ESA WorldCover 10 m 2021 v200*, DOI: https://doi.org/10.5281/zenodo.7254221
- Trang truy cập và attribution chính thức: https://esa-worldcover.org/en/data-access
- WorldCover download helper gốc: https://github.com/ESA-WorldCover/esa-worldcover-datasets
- Copernicus DEM public AWS readme: https://copernicus-dem-30m.s3.amazonaws.com/readme.html

Attribution cho bản đồ WorldCover: `© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium`.
