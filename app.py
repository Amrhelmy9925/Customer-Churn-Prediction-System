from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.prediction import router as prediction_router

# 1. Initialize the FastAPI application
app = FastAPI(title="Customer Churn Prediction App")

# 2. Mount static files directory
# This allows FastAPI to serve CSS, JS, and image files.
# We mount the 'static' folder at the '/static' URL path.
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Setup Jinja2 Templates
# This tells FastAPI where to look for HTML files.
templates = Jinja2Templates(directory="templates")

# 4. Include API routers
# This registers all routes defined in the prediction router (e.g., /api/predict)
app.include_router(prediction_router)


# 4. Define routes for each page
@app.get("/")
async def home(request: Request):
    """Serves the dashboard (index.html)"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/prediction")
async def prediction_page(request: Request):
    """Serves the prediction form page"""
    return templates.TemplateResponse("prediction.html", {"request": request})


@app.get("/performance")
async def performance_page(request: Request):
    """Serves the model performance metrics page"""
    return templates.TemplateResponse("performance.html", {"request": request})


@app.get("/features")
async def features_page(request: Request):
    """Serves the feature importance page"""
    return templates.TemplateResponse("features.html", {"request": request})


@app.get("/history")
async def history_page(request: Request):
    """Serves the prediction history page"""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/api/features")
async def get_features():
    """
    Returns feature importances from the trained model.
    Used by the Features page to render the importance chart.
    """
    from api.model_loader import get_models

    model, _, columns = get_models()

    # Extract importances (works for tree-based models like GradientBoosting)
    importances = model.feature_importances_

    # Pair columns with importances and sort descending
    features = sorted(
        [
            {"feature": col, "importance": float(imp)}
            for col, imp in zip(columns, importances)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return {"features": features}
