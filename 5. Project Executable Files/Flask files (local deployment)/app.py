from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load models
svm_model = pickle.load(open('svm_liver_analysis.pkl', 'rb'))
rfc_model = pickle.load(open('rfc_liver_analysis.pkl', 'rb'))
knn_model = pickle.load(open('knn_liver_analysis.pkl', 'rb'))
lr_model = pickle.load(open('lr_liver_analysis.pkl', 'rb'))

@app.route('/')
def loadpage():
    return render_template("LiverPredict.html")

@app.route('/y_predict', methods=["POST"])
def prediction():
    try:
        # Extracting input values from the form
        Gender = request.form.get("Gender")
        Age = request.form.get("Age")
        Total_Bilirubin = request.form.get("Total_Bilirubin")
        Direct_Bilirubin = request.form.get("Direct_Bilirubin")
        Alkaline_Phosphotase = request.form.get("Alkaline_Phosphotase")
        Alamine_Aminotransferase = request.form.get("Alamine_Aminotransferase")
        Aspartate_Aminotransferase = request.form.get("Aspartate_Aminotransferase")
        Total_Proteins = request.form.get("Total_Proteins")
        Albumin = request.form.get("Albumin")
        Albumin_and_Globulin_Ratio = request.form.get("Albumin_and_Globulin_Ratio")

        # Debugging: print received form data
        print(f"Received data: Gender={Gender}, Age={Age}, Total_Bilirubin={Total_Bilirubin}, Direct_Bilirubin={Direct_Bilirubin}, Alkaline_Phosphotase={Alkaline_Phosphotase}, Alamine_Aminotransferase={Alamine_Aminotransferase}, Aspartate_Aminotransferase={Aspartate_Aminotransferase}, Total_Proteins={Total_Proteins}, Albumin={Albumin}, Albumin_and_Globulin_Ratio={Albumin_and_Globulin_Ratio}")

        # Encoding gender
        gender_dict = {'Male': 1, 'Female': 0}
        Gender_encoded = gender_dict.get(Gender)

        # Check if Gender is valid
        if Gender_encoded is None:
            return render_template("LiverPredict.html", prediction_text="Invalid Gender input")

        # Creating a data array for prediction
        data = [[float(Gender_encoded), float(Age), float(Total_Bilirubin), float(Direct_Bilirubin),
                 float(Alkaline_Phosphotase), float(Alamine_Aminotransferase),
                 float(Aspartate_Aminotransferase), float(Total_Proteins), float(Albumin),
                 float(Albumin_and_Globulin_Ratio)]]
        data = np.array(data).reshape(1, -1)

        # Choose model to predict (example: SVM)
        model = svm_model
        prediction = model.predict(data)[0]

        # Generating prediction result text
        if prediction == 1:
            result_text = "You have a liver disease problem, you should see a doctor."
        else:
            result_text = "You do not have a liver disease problem."

        return render_template("LiverPredict.html", prediction_text=result_text)
    except Exception as e:
        return render_template("LiverPredict.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
