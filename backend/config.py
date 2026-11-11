"""UDI configuration.

All secrets come from the environment (or backend/.env, which is never packaged).
"""
from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Gemini ---------------------------------------------------------------
# Gemini is used ONLY for (a) turning a question into a structured plan and
# (b) wording an explanation of an already-verified Python result.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
# Up to N sample rows are included in the planning prompt (schema + stats are always sent,
# the full dataset never is). Set to 0 to send no row-level values at all.
GEMINI_SAMPLE_ROWS = int(os.getenv("GEMINI_SAMPLE_ROWS", "3"))

# --- Planner (Phase 2) --------------------------------------------------------
# auto: Gemini when configured, else (or on Gemini failure) the trained local intent model.
# gemini: Gemini only (no fallback). local: offline local planner only.
PLANNER_MODE = os.getenv("UDI_PLANNER", "auto").strip().lower()

# --- Server ----------------------------------------------------------------
PORT = int(os.getenv("PORT") or os.getenv("FLASK_PORT", "5000"))      # PORT is set by hosting platforms (Render)
MAX_UPLOAD_MB = int(os.getenv("UDI_MAX_UPLOAD_MB", "50"))
SESSION_TTL_SECONDS = int(os.getenv("UDI_SESSION_TTL_SECONDS", str(6 * 3600)))
MAX_SESSIONS = int(os.getenv("UDI_MAX_SESSIONS", "25"))

# --- Phase 2: persistence, security, limits ---------------------------------
DATA_DIR = Path(os.getenv("UDI_DATA_DIR", str(BASE_DIR / "data")))      # SQLite + dataset snapshots (never packaged)
PERSIST_SESSIONS = _bool("UDI_PERSIST_SESSIONS", True)
AUTH_ENABLED = _bool("UDI_AUTH", False)                                 # on: users must register / sign in
RATE_LIMITS = {"chat": int(os.getenv("UDI_RATE_CHAT", "40")), "upload": int(os.getenv("UDI_RATE_UPLOAD", "15")),
               "report": int(os.getenv("UDI_RATE_REPORT", "8")), "auth": int(os.getenv("UDI_RATE_AUTH", "20"))}
JOB_WORKERS = int(os.getenv("UDI_JOB_WORKERS", "2"))
CORS_ORIGINS = os.getenv("UDI_CORS_ORIGINS", "*")
VERSION = "2.0.0"

# --- Paths -----------------------------------------------------------------
SAMPLE_DIR = BASE_DIR / "sample_data"
REAL_SAMPLE_DIR = SAMPLE_DIR / "real"          # public real-world datasets used in the Phase 2 evaluation
UPLOAD_DIR = BASE_DIR / "uploads"          # temporary storage for uploaded files (never packaged)
MODEL_DIR = BASE_DIR / "models"          # domain *reference* models (validation / demo only)
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

# --- Seven contextual domains ------------------------------------------------
# A domain is CONTEXT: it tells Gemini which vocabulary the user is probably using.
# It never selects a model and never fixes a target. The uploaded dataset decides what is possible.
DOMAINS = {
    "company": {
        "label": "Company", "icon": "▣",
        "description": "Sales, demand, inventory, customers and profitability.",
        "context": "A commercial organisation: revenue, sales, cost, profit, demand, inventory, customers, employees, products, regions.",
    },
    "hospital": {
        "label": "Hospital", "icon": "✚",
        "description": "Patient load, capacity, waiting time and staffing.",
        "context": "A hospital or clinic: patients, admissions, departments, beds, waiting time, staffing, treatments, outcomes.",
    },
    "college": {
        "label": "College", "icon": "⌂",
        "description": "Attendance, students, faculty and resource utilization.",
        "context": "A college or university: students, attendance, faculty, departments, courses, grades, classroom utilisation.",
    },
    "municipality": {
        "label": "Municipality", "icon": "◈",
        "description": "Waste, complaints, service demand and zones.",
        "context": "A city or local government: zones/wards, waste collection, complaints, service requests, workers, population.",
    },
    "ngo": {
        "label": "NGO", "icon": "♡",
        "description": "Beneficiaries, donations, aid and volunteer demand.",
        "context": "A non-profit: beneficiaries, donations, aid distribution, volunteers, programmes, districts.",
    },
    "community": {
        "label": "Community", "icon": "⌁",
        "description": "Local requests, participation, shortages and needs.",
        "context": "A local community body: resource requests, participation, shortages, areas, needs.",
    },
    "event_organizer": {
        "label": "Event Organizer", "icon": "★",
        "description": "Attendance, capacity, registrations and resources.",
        "context": "An event business: attendance, capacity, registrations, ticket revenue, resources, event types.",
    },
}


# Bundled datasets the UI can load with one click (generated reference data + public real-world data).
REAL_DATASETS = {
    "global_superstore_sample": {"label": "Global Superstore (sample)", "domain": "company", "file": "global_superstore_sample.csv",
                                 "source": "Global Superstore retail orders (public Tableau sample), 1,000-row extract", "use": "sales, operations, finance"},
    "ibm_hr_attrition": {"label": "IBM HR Attrition", "domain": "company", "file": "ibm_hr_attrition.csv",
                         "source": "IBM HR Analytics Employee Attrition & Performance (IBM, public)", "use": "HR"},
    "ibm_telco_churn": {"label": "IBM Telco Customer Churn", "domain": "company", "file": "ibm_telco_churn.csv",
                        "source": "IBM Telco Customer Churn (IBM, public)", "use": "marketing, customer"},
    "uci_bank_marketing": {"label": "UCI Bank Marketing", "domain": "company", "file": "uci_bank_marketing.csv",
                           "source": "Moro et al., UCI Bank Marketing (bank-additional-full)", "use": "marketing"},
}
