# Main FastAPI app entrypoint to include placeholder and existing routers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import placeholder_routes
from api.alerts import router as alerts_router
from api.performance import router as performance_router

app = FastAPI()

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include placeholder API routes
app.include_router(placeholder_routes.router)
# Include real alerts API endpoints
app.include_router(alerts_router)
# Include real performance API endpoints
app.include_router(performance_router)

@app.get("/")
def root():
    return {"status": "API running"}
