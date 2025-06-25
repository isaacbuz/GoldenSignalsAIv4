# Main FastAPI app entrypoint to include placeholder and existing routers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import placeholder_routes
from src.api.alerts import router as alerts_router
from src.api.performance import router as performance_router
from src.api.options_route import router as options_route_router
from src.api.etf_heatmap import router as etf_heatmap_router
from src.api.signal_cluster_stats import router as signal_cluster_stats_router
from src.api.analytics import router as analytics_router
from src.api.arbitrage import router as arbitrage_router

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
# Include analytics API endpoints
app.include_router(analytics_router)
# Include options flow API endpoints
app.include_router(options_route_router)
# Include ETF heatmap API endpoints
app.include_router(etf_heatmap_router)
# Include signal cluster stats API endpoints
app.include_router(signal_cluster_stats_router)
# Include arbitrage API endpoints
app.include_router(arbitrage_router)

@app.get("/")
def root():
    return {"status": "API running"}
