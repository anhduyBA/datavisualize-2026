import seaborn as sns

df = sns.load_dataset("titanic")

print(df.head())
# Hiển thị 5 dòng đầu tiên của dataset Titanic.
print(df.info)
# In ra object thông tin về phương thức df.info (chưa gọi hàm).

print(df.isnull().sum())
# Hiển thị tổng số giá trị null theo mỗi cột.

null_counts = df.isnull().sum()
print(null_counts.idxmax())
# ## null_counts.idxmax()
# Trả về tên cột có nhiều giá trị null nhất.


print(df.describe())
# ## df.describe()
# Hiển thị thống kê mô tả (count, mean, std, min, max, percentiles) cho các cột số.
print(df["age"].mean())
# ## df["age"].mean()
# Hiển thị tuổi trung bình của các hành khách.
print(df["fare"].max())
# ## df["fare"].max()
# Hiển thị giá vé cao nhất.

print(df["sex"].value_counts())
# ## df["sex"].value_counts()
# Đếm số lượng hành khách theo giới tính.

print(df["sex"].value_counts())
# ## df["sex"].value_counts() (lặp lại)
# Lần thứ hai in lại số lượng hành khách theo giới tính.

print(df["pclass"].value_counts())
# ## df["pclass"].value_counts()
# Đếm số lượng hành khách theo hạng vé.

print(df["pclass"].value_counts(normalize=True).round(3) * 100)
# ## df["pclass"].value_counts(normalize=True).round(3) * 100
# Hiển thị tỷ lệ phần trăm hành khách theo hạng vé, làm tròn 3 chữ số.

women = df[df["sex"] == "female"]
pct = len(women) / len(df) * 100
print("Female percent:", pct)
# ## Female percent
# In phần trăm hành khách là nữ trong tổng dataset.

first = df[df["pclass"] == 1]
print("Mean fare in class 1:", first["fare"].mean())
# ## Mean fare in class 1
# Trung bình giá vé của hành khách hạng 1.

print("Count age < 18:", df[df["age"] < 18].shape[0])
# ## Count age < 18
# Đếm số hành khách dưới 18 tuổi.

print("Count age > 60:", df[df["age"] > 60].shape[0])
# ## Count age > 60
# Đếm số hành khách trên 60 tuổi.


# Lưu ý: age có null – dropna hoặc dùng .notna()

#T3

# =========================================
# PHẦN A — ĐẶT CÂU HỎI TRƯỚC KHI CODE
# =========================================

# Câu 1: Tỷ lệ sống sót trên Titanic là bao nhiêu?
# Tôi dự đoán kết quả sẽ là: số người không sống sót nhiều hơn số người sống sót.

# Câu 2: Giới tính nào có tỷ lệ sống sót cao hơn?
# Tôi dự đoán kết quả sẽ là: nữ sống sót nhiều hơn nam.

# Câu 3: Hạng vé nào có tỷ lệ sống sót cao nhất?
# Tôi dự đoán kết quả sẽ là: hành khách hạng 1 sống sót cao nhất.

# Câu 4: Người lớn tuổi có sống sót ít hơn không?
# Tôi dự đoán kết quả sẽ là: người lớn tuổi sống sót ít hơn.

# Câu 5: Trẻ em có tỷ lệ sống sót cao hơn người lớn không?
# Tôi dự đoán kết quả sẽ là: trẻ em có tỷ lệ sống sót cao hơn.

# Câu 6: Giá vé càng cao thì khả năng sống sót có cao hơn không?
# Tôi dự đoán kết quả sẽ là: người trả vé cao có tỷ lệ sống sót cao hơn.

# Câu 7: Có bao nhiêu dữ liệu bị thiếu trong dataset?
# Tôi dự đoán kết quả sẽ là: cột age có nhiều giá trị thiếu.

# Câu 8: Hành khách nam hay nữ nhiều hơn?
# Tôi dự đoán kết quả sẽ là: số nam nhiều hơn nữ.

# Câu 9: Hạng vé nào có nhiều hành khách nhất?
# Tôi dự đoán kết quả sẽ là: hạng 3 nhiều hành khách nhất.

# Câu 10: Độ tuổi trung bình của hành khách khoảng bao nhiêu?
# Tôi dự đoán kết quả sẽ là: khoảng 28–30 tuổi.


# =========================================
# IMPORT DATASET
# =========================================

import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")


# =========================================
# PHẦN B — KHÁM PHÁ CẤU TRÚC DỮ LIỆU
# =========================================

print("===== HEAD =====")
print(df.head())

# Markdown giải thích:
# head() hiển thị 5 dòng đầu tiên của dataset.
# Qua kết quả có thể thấy dataset chứa thông tin về hành khách như tuổi, giới tính, hạng vé,...
# Đây là bước đầu để quan sát cấu trúc dữ liệu.


print("\n===== TAIL =====")
print(df.tail())

# Markdown giải thích:
# tail() hiển thị 5 dòng cuối cùng của dataset.
# Giúp kiểm tra dữ liệu ở cuối bảng có bị lỗi hoặc thiếu dữ liệu không.


print("\n===== INFO =====")
print(df.info())

# Markdown giải thích:
# info() hiển thị số dòng, số cột và kiểu dữ liệu.
# Có thể thấy dataset có cả numeric và categorical data.
# Một số cột có ít dữ liệu hơn tổng số dòng, nghĩa là có missing values.


print("\n===== SHAPE =====")
print(df.shape)

# Markdown giải thích:
# shape cho biết số dòng và số cột của dataset.
# Titanic dataset có 891 dòng dữ liệu.


print("\n===== DESCRIBE =====")
print(df.describe())

# Markdown giải thích:
# describe() hiển thị thống kê mô tả cho các cột số.
# Có thể quan sát mean, min, max và các giá trị bất thường.
# Ví dụ fare có giá trị max rất lớn so với trung bình.


# =========================================
# KIỂM TRA MISSING VALUES
# =========================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Markdown giải thích:
# isnull().sum() dùng để đếm số lượng giá trị null theo từng cột.
# age, deck và embarked có dữ liệu bị thiếu.
# deck có số lượng thiếu rất lớn nên có thể khó sử dụng để phân tích.


null_counts = df.isnull().sum()

print("\nCột có nhiều giá trị null nhất:")
print(null_counts.idxmax())


# =========================================
# PHÂN TÍCH CATEGORICAL COLUMNS
# sex, pclass, embarked
# =========================================

print("\n===== SEX VALUE COUNTS =====")
print(df["sex"].value_counts())

# Markdown giải thích:
# Dataset có số lượng nam nhiều hơn nữ.
# Điều này có thể ảnh hưởng đến tỷ lệ sống sót chung.


print("\n===== PCLASS VALUE COUNTS =====")
print(df["pclass"].value_counts())

# Markdown giải thích:
# Hành khách hạng 3 chiếm số lượng lớn nhất.
# Hạng 1 có ít hành khách hơn nhưng giá vé cao hơn.


print("\n===== EMBARKED VALUE COUNTS =====")
print(df["embarked"].value_counts())

# Markdown giải thích:
# Phần lớn hành khách lên tàu tại cảng Southampton (S).
# Một số ít lên tàu tại Queenstown (Q).


# =========================================
# PHÂN TÍCH NUMERIC COLUMNS
# age, fare, sibsp
# =========================================

print("\n===== AGE STATS =====")
print("Mean:", df["age"].mean())
print("Median:", df["age"].median())
print("Min:", df["age"].min())
print("Max:", df["age"].max())

# Markdown giải thích:
# Tuổi trung bình khoảng 29 tuổi.
# Có hành khách rất nhỏ tuổi và cũng có người lớn tuổi.
# Dữ liệu age có missing values nên mean có thể chưa hoàn toàn chính xác.


print("\n===== FARE STATS =====")
print("Mean:", df["fare"].mean())
print("Median:", df["fare"].median())
print("Min:", df["fare"].min())
print("Max:", df["fare"].max())

# Markdown giải thích:
# Giá vé cao nhất lớn hơn rất nhiều so với trung bình.
# Điều này cho thấy có outlier trong cột fare.


print("\n===== SIBSP STATS =====")
print("Mean:", df["sibsp"].mean())
print("Median:", df["sibsp"].median())
print("Min:", df["sibsp"].min())
print("Max:", df["sibsp"].max())

# Markdown giải thích:
# Phần lớn hành khách đi một mình hoặc có ít người thân đi cùng.
# Có một vài trường hợp số lượng người thân rất lớn.


# =========================================
# PHẦN C — PHÂN TÍCH CƠ BẢN
# =========================================

# Tỷ lệ sống sót toàn tàu
print("\n===== SURVIVAL RATE =====")

survival_rate = df["survived"].mean() * 100
print("Survival rate:", round(survival_rate, 2), "%")

# Markdown giải thích:
# Tỷ lệ sống sót được tính bằng trung bình của cột survived.
# Kết quả cho thấy số người không sống sót nhiều hơn số sống sót.


# =========================================
# GROUPBY THEO GIỚI TÍNH
# =========================================

print("\n===== SURVIVAL BY SEX =====")

sex_survival = df.groupby("sex")["survived"].mean() * 100
print(sex_survival)

# Markdown giải thích:
# Nữ có tỷ lệ sống sót cao hơn nam rất nhiều.
# Điều này phù hợp với quy tắc ưu tiên phụ nữ và trẻ em.


# =========================================
# GROUPBY THEO HẠNG VÉ
# =========================================

print("\n===== SURVIVAL BY PCLASS =====")

pclass_survival = df.groupby("pclass")["survived"].mean() * 100
print(pclass_survival)

# Markdown giải thích:
# Hành khách hạng 1 có tỷ lệ sống sót cao nhất.
# Hành khách hạng 3 có tỷ lệ sống sót thấp nhất.


# =========================================
# TRẢ LỜI ÍT NHẤT 3 CÂU HỎI PHẦN A
# =========================================

# Câu hỏi 1
print("\n===== QUESTION 1 =====")
print("Tỷ lệ sống sót:", round(survival_rate, 2), "%")

# Nhận xét:
# Kết quả đúng với dự đoán:
# số người không sống sót nhiều hơn số người sống sót.


# Câu hỏi 2
print("\n===== QUESTION 2 =====")
print(sex_survival)

# Nhận xét:
# Kết quả đúng với dự đoán:
# nữ có tỷ lệ sống sót cao hơn nam.


# Câu hỏi 3
print("\n===== QUESTION 3 =====")
print(pclass_survival)

# Nhận xét:
# Kết quả đúng với dự đoán:
# hạng 1 có tỷ lệ sống sót cao nhất.


# =========================================
# ĐỐI CHIẾU DỰ ĐOÁN VS THỰC TẾ
# =========================================

print("\n===== PREDICTION VS REALITY =====")

print("1. Người sống sót ít hơn người chết -> ĐÚNG")
print("2. Nữ sống sót nhiều hơn nam -> ĐÚNG")
print("3. Hạng 1 sống sót cao nhất -> ĐÚNG")

# Markdown nhận xét:
# Phần lớn các dự đoán ban đầu đều đúng.
# Điều này cho thấy dữ liệu Titanic có xu hướng khá rõ ràng theo giới tính và hạng vé.
# Các yếu tố như giới tính và điều kiện kinh tế có ảnh hưởng mạnh đến khả năng sống sót.


# =========================================
# PHẦN D — CÂU HỎI MỚI SAU PHÂN TÍCH
# =========================================

# 1. Người đi một mình có sống sót nhiều hơn người đi cùng gia đình không?
# 2. Người trả vé cao nhất thuộc hạng vé nào?
# 3. Cảng lên tàu nào có tỷ lệ sống sót cao nhất?
# 4. Tuổi trung bình của người sống sót là bao nhiêu?
# 5. Nam ở hạng 1 có tỷ lệ sống sót cao hơn nữ ở hạng 3 không?


# =========================================
# BONUS — TRẢ LỜI 2 CÂU HỎI MỚI
# =========================================

# Câu bonus 1
print("\n===== BONUS 1 =====")

embarked_survival = df.groupby("embarked")["survived"].mean() * 100
print(embarked_survival)

# Nhận xét:
# Một số cảng có tỷ lệ sống sót cao hơn các cảng khác.
# Điều này có thể liên quan đến hạng vé của hành khách tại từng cảng.


# Câu bonus 2
print("\n===== BONUS 2 =====")

survived_age_mean = df[df["survived"] == 1]["age"].mean()
print("Mean age of survived passengers:", survived_age_mean)

# Nhận xét:
# Tuổi trung bình của người sống sót thấp hơn một chút so với toàn bộ dataset.
# Điều này cho thấy trẻ em và người trẻ có thể được ưu tiên cứu hộ hơn.