# 🎓 Student Performance Classification

A machine learning project to predict student pass/fail outcomes based on various academic, demographic, and behavioral factors. Built with Python, scikit-learn, and Streamlit.

---

## 🚀 Project Overview

This project uses a dataset of student performance factors to train a classifier that predicts whether a student will "pass" or "fail" an exam. The model leverages features such as hours studied, attendance, parental involvement, motivation level, and more.

> **Disclaimer:**  
> Due to the absence of failing students (Exam_Score < 35) in the available dataset, we define "Pass" as `Exam_Score >= 70` and "Fail" as `< 70` for demonstration purposes. This threshold does **not** reflect the real-world academic pass/fail standard.


## 📂 Project Structure

![image](https://github.com/user-attachments/assets/9ae80e7c-9a64-4e08-9f73-6948e3ab2564)


---

## 🛠️ Features

- Data preprocessing (missing values, encoding, scaling)
- Feature engineering and selection
- Model training (Random Forest Classifier)
- Interactive prediction app via Streamlit
- Model persistence with joblib

---

## 🏁 How to Run

1. **Clone the repository**
Bach
    git clone https://github.com/Harshkuamr00/Student_Performance_analysis.git

    cd Student_Performance_analysis

2. **Install dependencies**
    Refer the requirement.txt

3. **Train the model**
   Run the notebook or script to preprocess data and train the model:
    ```
    python3 st.ipynb
    ```
    or run the Jupyter notebook.

4. **Start the Streamlit app**
Bach
    streamlit run app.py
   

5. **Use the sidebar to enter student feature values and get a Pass/Fail prediction!**

---

## 📊 Example Features

- Hours Studied
- Attendance
- Parental Involvement
- Access to Resources
- Motivation Level
- Previous Scores
- Tutoring Sessions
- Family Income
- Teacher Quality
- School Type
- Peer Influence
- Physical Activity
- Learning Disabilities
- Parental Education Level
- Distance from Home
- Gender

---

## ⚠️ Target Definition

**Important:**  
- The original dataset did **not** contain any students with `Exam_Score < 35` (all students "passed").
- To enable binary classification, we set the "Pass" threshold at 70:
- **Pass:** `Exam_Score >= 70`
- **Fail:** `Exam_Score < 70`
- This is for demonstration purposes only and does **not** reflect actual academic standards.

---

## 📈 Results

- Model: Random Forest Classifier
- Accuracy: _[0.9220877458396369]_

---

## 📚 Requirements

- Python 3.7+
- pandas
- scikit-learn
- streamlit
- joblib

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [UCI Student Performance Dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- scikit-learn, pandas, Streamlit

---






