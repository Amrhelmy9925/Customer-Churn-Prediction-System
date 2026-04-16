# 🚀 Customer Churn Prediction System

An AI-powered analytics dashboard designed to predict customer churn with high accuracy. Built with **FastAPI** and **Scikit-learn**, this application provides real-time insights, interactive visualizations, and a seamless user experience for data-driven decision-making.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| 📊 **Interactive Dashboard** | Real-time overview of customer stats, churn rates, and model performance. |
| 🧠 **Smart Predictions** | Instant churn probability analysis with risk classification (Low/Medium/High). |
| 🔀 **Shuffle Mode** | Generate realistic, randomized customer profiles for rapid testing. |
| 📈 **Feature Importance** | Visual breakdown of the top factors driving customer churn. |
| 📜 **Prediction History** | Track, filter, and export past predictions to CSV. |
| 🐳 **Dockerized** | Production-ready containerization for easy deployment. |
| 📱 **Responsive UI** | Modern, mobile-friendly interface built with clean CSS3. |

---

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic](https://docs.pydantic.dev/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript, [FontAwesome](https://fontawesome.com/)
- **Containerization**: [Docker](https://www.docker.com/)

---

## 📂 Project Structure

```text
pr/
├── api/                 # 📦 API Logic
│   ├── __init__.py
│   ├── model_loader.py  # Model loading & caching
│   └── prediction.py    # Prediction endpoints & validation
├── models/              # 🧠 Trained ML Artifacts
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── training_columns.pkl
├── src/                 # 📚 Training Scripts (Reference)
├── templates/           # 🎨 Frontend Assets
│   ├── index.html       # Dashboard
│   ├── prediction.html  # Prediction Form
│   ├── performance.html # Model Metrics
│   ├── features.html    # Feature Importance
│   ├── history.html     # History & Export
├── static/
│   ├── styles.css       # Global Styles
│   └── app.js           # Client-side Logic
├── app.py               # 🚀 Main FastAPI Entry Point
├── requirements.txt     # 📋 Python Dependencies
├── Dockerfile           # 🐳 Container Definition
└── README.md            # 📖 Documentation
```

---

## 🚀 Getting Started

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pr
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

5. **Open in browser**
   Navigate to [http://localhost:8000](http://localhost:8000)

### Option 2: Docker (Recommended)

1. **Build the image**
   ```bash
   docker build -t churn-predictor .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 --name churn-app churn-predictor
   ```

3. **Access the app**
   Open [http://localhost:8000](http://localhost:8000)

---

## 🔌 API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, visit:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Key Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Dashboard Home |
| `GET` | `/prediction` | Prediction Interface |
| `POST` | `/api/predict` | Submit data for churn prediction |
| `GET` | `/api/features` | Retrieve feature importance data |

---

## 🧠 Model Details

- **Algorithm**: Gradient Boosting Classifier
- **Performance**: ROC AUC ~0.8465
- **Features**: 30 (One-hot encoded categorical + numerical)
- **Preprocessing**: StandardScaler for numerical features

---

## 📸 Usage

1. **Navigate to Prediction**: Click "Prediction" in the sidebar.
2. **Shuffle Data**: Click the **🔀 Shuffle** button to auto-fill the form with a random customer profile.
3. **Predict**: Click **Predict Churn** to see the probability and risk level.
4. **View History**: Check the "History" tab to see past predictions or export them as CSV.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ by [Your Name]**
