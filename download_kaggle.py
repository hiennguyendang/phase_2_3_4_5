import kagglehub

# KEY: KGAT_6e4e483f8273bee6280ec061fd0d33ac

# 1. Đăng nhập qua terminal (chỉ cần thiết nếu dataset là private)
kagglehub.login()

# 2. Tải toàn bộ dataset thẳng vào thư mục ./data
path = kagglehub.dataset_download(
    "nguynnghin/frozen", 
    output_dir="./data/features/frozen"
)

print("Hoàn tất! Đường dẫn dataset của bạn là:", path)