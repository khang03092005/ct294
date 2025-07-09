from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

from feature_advice import feature_advice  # Load mô tả

app = Flask(__name__)

feature_names = ['Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
    'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
    'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology']
label_mapping = {
    "Age": "Tuổi",
    "Number of sexual partners": "Số bạn tình",
    "First sexual intercourse": "Tuổi quan hệ lần đầu",
    "Num of pregnancies": "Số lần mang thai",
    "Smokes": "Hút thuốc",
    "Smokes (years)": "Số năm hút thuốc",
    "Smokes (packs/year)": "Số gói mỗi năm",
    "Hormonal Contraceptives": "Dùng thuốc tránh thai",
    "Hormonal Contraceptives (years)": "Số năm dùng thuốc tránh thai",
    "IUD": "Đặt vòng",
    "IUD (years)": "Số năm đặt vòng",
    "STDs": "Từng mắc STDs",
    "STDs (number)": "Số lần mắc STDs",
    "STDs:condylomatosis": "Sùi mào gà",
    "STDs:cervical condylomatosis": "Sùi mào gà cổ tử cung",
    "STDs:vaginal condylomatosis": "Sùi mào gà âm đạo",
    "STDs:vulvo-perineal condylomatosis": "Sùi mào gà âm hộ - tầng sinh môn",
    "STDs:syphilis": "Bệnh giang mai",
    "STDs:pelvic inflammatory disease": "Viêm vùng chậu",
    "STDs:genital herpes": "Mụn rộp sinh dục",
    "STDs:molluscum contagiosum": "U mềm lây",
    "STDs:AIDS": "AIDS",
    "STDs:HIV": "HIV",
    "STDs:Hepatitis B": "Viêm gan B",
    "STDs:HPV": "Nhiễm HPV",
    "STDs: Number of diagnosis": "Số lần được chẩn đoán STDs",
    "STDs: Time since first diagnosis": "Thời gian từ lần chẩn đoán STDs đầu tiên",
    "STDs: Time since last diagnosis": "Thời gian từ lần chẩn đoán STDs gần nhất",
    "Dx:Cancer": "Chẩn đoán ung thư",
    "Dx:CIN": "Chẩn đoán CIN",
    "Dx:HPV": "Chẩn đoán HPV",
    "Dx": "Chẩn đoán bất thường",
    "Hinselmann": "Hinselmann dương tính",
    "Schiller": "Schiller dương tính",
    "Citology": "Tế bào học dương tính",
    "Biopsy": "Sinh thiết dương tính"
}


model = joblib.load("logistic_model.pkl")
X_background = shap.maskers.Independent(pd.DataFrame([[0]*len(feature_names)], columns=feature_names))
explainer = shap.LinearExplainer(model, masker=X_background, feature_names=feature_names)


def generate_advice_auto(name, value, shap_val, percent):
    trend = "tăng nguy cơ" if shap_val > 0 else "giảm nguy cơ"
    vi_name = label_mapping.get(name, name)
    line = f"• {vi_name} = {value} → {trend} ({shap_val:+.2f}, ảnh hưởng: {percent:.1f}%)\n"


    if name in feature_advice:
        desc = feature_advice[name]["desc"]
        action = feature_advice[name]["action"]
    else:
        desc = f"Yếu tố này ảnh hưởng {trend} đến nguy cơ mắc bệnh."
        action = "Nên tham khảo ý kiến bác sĩ nếu bạn chưa rõ về yếu tố này."

    return f"{line}  {desc}\n  👉 {action}\n"

@app.route("/",methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in feature_names]
            X_input = pd.DataFrame([values], columns=feature_names)

            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1] * 100

            # SHAP
            shap_values = explainer(X_input)
            shap_score = shap_values.values[0]
            total_abs = sum(abs(val) for val in shap_score)
            impacts = [
                (feature_names[i], shap_score[i], abs(shap_score[i]) / total_abs * 100)
                for i in range(len(feature_names))
            ]

            advice = ""
            filtered = [x for x in impacts if x[2] >= 5]
            if prediction == 1:
                if filtered:
                    advice += "🧠 Các yếu tố ảnh hưởng lớn đến dự đoán:\n\n"
                    for name, shap_val, percent in filtered:
                        idx = feature_names.index(name)
                        val = values[idx]
                        advice += generate_advice_auto(name, val, shap_val, percent)
                else:
                    advice = "⚠️ Nguy cơ cao nhưng không có yếu tố nào vượt ngưỡng 10% ảnh hưởng."
            else:
                advice = (
                    "✅ Bạn hiện không có nguy cơ đáng kể.\n\n"
                    "💡 Tuy nhiên, hãy:\n"
                    "• Duy trì lối sống lành mạnh\n"
                    "• Khám phụ khoa định kỳ (ít nhất mỗi 6–12 tháng)\n"
                    "• Tránh hút thuốc, hạn chế rượu bia\n"
                    "• Nếu chưa tiêm vaccine HPV, nên tham khảo ý kiến bác sĩ về việc tiêm phòng\n"
                    "\nChúc bạn luôn khỏe mạnh ❤️"
                )

           
            # Lưu biểu đồ waterfall SHAP
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig("static/shap_plot.png", bbox_inches='tight')
            plt.close()

            return render_template("index.html", features=feature_names,
                                   result=prediction, proba=round(proba, 2),
                                   advice=advice)
        except Exception as e:
            return f"Lỗi xử lý dữ liệu: {e}"
    return render_template("index.html", features=feature_names, result=None)
if __name__ == "__main__":
    app.run(debug=True)