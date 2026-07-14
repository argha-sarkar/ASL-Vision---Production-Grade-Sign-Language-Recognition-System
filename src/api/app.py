"""
app.py

FastAPI Application

Author: Argha Sarkar Project
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.routes import router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(
    router,
    prefix=settings.API_PREFIX,
    tags=["ASL Vision"],
)

# ---------------------------------------------------------
# Startup Event
# ---------------------------------------------------------


@app.on_event("startup")
async def startup():

    print("\n" + "=" * 70)
    print("ASL Vision API Started")
    print("=" * 70)

    print(f"Application : {settings.APP_NAME}")

    print(f"Version     : {settings.VERSION}")

    print(f"Host        : {settings.HOST}")

    print(f"Port        : {settings.PORT}")

    print("=" * 70)


# ---------------------------------------------------------
# Shutdown Event
# ---------------------------------------------------------


@app.on_event("shutdown")
async def shutdown():

    print("\nShutting down ASL Vision API...\n")


# ---------------------------------------------------------
# Run Application
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
