import numpy as np

# --- 1. VERİ TANIMI (Terminal Çıktınızdan Alınmıştır) ---

all_results = [
    {"tag": "S_MSE_NoBias", "B": 1, "train_acc": 0.6027, "test_acc": 0.6162, "total_train_time": 1656.65, "avg_epoch_time": 23.7736, "inf_time_per_sample": 4.28e-05},
    {"tag": "S_MSE_Bias", "B": 1, "train_acc": 0.6362, "test_acc": 0.6467, "total_train_time": 1051.75, "avg_epoch_time": 13.7474, "inf_time_per_sample": 4.59e-05},
    {"tag": "SM_CE_NoBias", "B": 1, "train_acc": 0.1124, "test_acc": 0.1135, "total_train_time": 970.55, "avg_epoch_time": 13.0598, "inf_time_per_sample": 4.21e-05},
    {"tag": "SM_CE_Bias", "B": 1, "train_acc": 0.1124, "test_acc": 0.1135, "total_train_time": 1024.06, "avg_epoch_time": 13.8145, "inf_time_per_sample": 4.42e-05},
    {"tag": "S_MSE_Bias", "B": 10, "train_acc": 0.8971, "test_acc": 0.9075, "total_train_time": 519.84, "avg_epoch_time": 4.9734, "inf_time_per_sample": 4.52e-05},
    {"tag": "S_MSE_Bias", "B": 100, "train_acc": 0.8741, "test_acc": 0.8859, "total_train_time": 521.72, "avg_epoch_time": 5.0132, "inf_time_per_sample": 4.53e-05},
]

# --- 2. TABLO OLUŞTURMA FONKSİYONU ---

def print_summary_table(results):
    # Başlıklar
    headers = ["Config (Tag)", "B", "Train Acc", "Test Acc", "Avg Epoch Time (s)", "Total Time (s)", "Inf Time (s)"]
    
    # Sütun genişlikleri (Genişliği en uzun başlığa göre ayarlar)
    widths = [20, 3, 10, 10, 20, 15, 15]

    # Başlıkları ve ayırıcıyı yazdırma
    header_line = f"{headers[0]:<{widths[0]}} | {headers[1]:<{widths[1]}} | {headers[2]:<{widths[2]}} | {headers[3]:<{widths[3]}} | {headers[4]:<{widths[4]}} | {headers[5]:<{widths[5]}} | {headers[6]:<{widths[6]}}"
    
    separator = "-" * (len(header_line) + 2)

    print("\n\n=== FINAL EXPERIMENTAL RESULTS SUMMARY ===")
    print(separator)
    print(header_line)
    print(separator)

    # Veri satırlarını yazdırma
    for res in results:
        row = (
            f"{res['tag']:<{widths[0]}} | "
            f"{res['B']:<{widths[1]}} | "
            f"{res['train_acc']:<{widths[2]}.4f} | "
            f"{res['test_acc']:<{widths[3]}.4f} | "
            f"{res['avg_epoch_time']:<{widths[4]}.4f} | "
            f"{res['total_train_time']:<{widths[5]}.2f} | "
            f"{res['inf_time_per_sample']:<{widths[6]}.2e}"
        )
        print(row)
    
    print(separator)

# Tabloyu oluştur ve yazdır
print_summary_table(all_results)