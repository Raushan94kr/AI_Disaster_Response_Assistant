# AI Disaster Response Assistant

## Building Damage Assessment Using Machine Learning and Computer Vision

### 1. Project Overview

The AI Disaster Response Assistant is a computer vision application designed to assess building damage from post-disaster satellite imagery. It uses building annotations from the xBD dataset and a trained MobileNetV2 classification model to categorize building damage.

The application provides a dashboard for before-and-after image comparison, building-level damage visualization, impact analytics, rescue priority assessment, and report generation.

### 2. Objectives

- Identify building locations using xBD building polygon annotations.
- Classify building damage from post-disaster imagery.
- Compare pre-disaster and post-disaster images.
- Visualize building-level predictions and damage categories.
- Present damage statistics and rescue-priority information.
- Generate reports for analysis and documentation.

### 3. Main Features

- Before-and-after disaster image comparison.
- Automatic matching of post-disaster images with xBD JSON annotations.
- Building polygon localization.
- MobileNetV2-based damage classification.
- Four damage classes: No Damage, Minor Damage, Major Damage, and Destroyed.
- Damage analytics and rescue-priority dashboard.
- CSV and PDF report generation.
- FastAPI backend and Streamlit frontend.

### 4. Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Backend:** FastAPI and Uvicorn
- **Machine Learning:** TensorFlow and MobileNetV2
- **Image Processing:** Pillow and NumPy
- **Dataset:** xBD building damage assessment dataset
- **Annotation Format:** JSON building polygons
- **Reporting:** ReportLab for PDF generation

### 5. Project Structure

```text
AI_Disaster_Response_Assistant/
├── app.py
├── backend/
│   ├── main.py
│   └── services/
│       └── analysis_service.py
├── models/
│   ├── class_names.json
│   └── disaster_damage_mobilenetv2.keras
├── dataset/
├── reports/
├── requirements.txt
├── requirements-full.txt
├── README.md
└── .gitignore
```

*Note: The structure shows the main project files. Additional files and dataset subfolders may vary.*

### 6. Installation and Setup

**Prerequisites**

- Python 3.12 or another compatible Python version.
- Git, if cloning the repository.

**Step 1: Clone the repository**

```bash
git clone https://github.com/Raushan94kr/AI_Disaster_Response_Assistant.git
cd AI_Disaster_Response_Assistant
```

The initial repository upload contains `README.md` and `app.py`. Before running
the application, also provide the supporting project files and assets described
in the project structure, including `requirements.txt`, the trained model at
`models/disaster_damage_mobilenetv2.keras`, and the backend files. xBD images
and annotations are needed for building-level analysis.

**Step 2: Create a virtual environment**

```powershell
python -m venv venv
```

**Step 3: Activate the virtual environment**

```powershell
.\venv\Scripts\Activate.ps1
```

**Step 4: Install dependencies**

```powershell
python -m pip install -r requirements.txt
```

**Step 5: Verify dependencies**

```powershell
python -m pip check
```

### 7. Running the Application

The application uses a FastAPI backend and a Streamlit frontend. Start each component in a separate terminal.

**Start the FastAPI backend**

```powershell
python -m uvicorn backend.main:app --reload
```

Backend health endpoint:

`http://127.0.0.1:8000/api/health`

**Start the Streamlit frontend**

```powershell
streamlit run app.py
```

Open the application in your browser:

`http://localhost:8501`

Keep both terminals running while using the application.

### 8. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check backend status |
| POST | `/api/analyze` | Analyze an uploaded post-disaster image |

The analysis endpoint supports JPG, JPEG, and PNG images. The expected image naming convention and matching JSON annotation requirements depend on the backend implementation and available dataset files.

### 9. Model Information

- **Model Architecture:** MobileNetV2
- **Input Size:** 224 × 224 pixels
- **Reported Test Accuracy:** 86%
- **Output Classes:** No Damage, Minor Damage, Major Damage, Destroyed
- **Building Localization:** xBD JSON polygon annotations

The reported accuracy is based on the existing project evaluation results. Actual performance may vary on unseen real-world disaster imagery.

### 10. Dataset and Processing Workflow

1. Select a pre-disaster and post-disaster image pair.
2. Load the post-disaster image and locate its matching xBD JSON annotation.
3. Extract building polygons from the annotation.
4. Process building image regions for classification.
5. Predict a damage category for each analyzed building.
6. Display building-level results, analytics, and rescue-priority information.
7. Generate reports using the available reporting features.

### 11. Limitations

- The model's predictions may be incorrect or uncertain.
- Performance can vary with image quality, disaster type, and unseen data.
- Matching JSON annotations must be available for supported post-disaster images.
- Rescue priorities are model-generated indicators and require human verification.
- The current application uses local services and is not automatically deployed as a public website.
- Predictions should support, not replace, professional disaster-response decisions.

### 12. Future Scope

- Evaluate performance on additional disaster datasets.
- Improve classification accuracy and robustness.
- Add interactive geographic mapping and geolocation features.
- Improve multilingual reporting and accessibility.
- Deploy the application on a suitable server with security controls.
- Evaluate prediction confidence and model limitations more thoroughly.

### 13. Conclusion

The AI Disaster Response Assistant combines satellite-image comparison, xBD building annotations, machine learning, and a web dashboard to support building damage assessment after disasters. It provides building-level visualization and analytical information that may help organize disaster-response activities. Further validation and field evaluation are needed before operational use.

### 14. Disclaimer

This application is an academic and technical project. Its predictions are not a substitute for verified field inspections or decisions by qualified emergency-response personnel.
