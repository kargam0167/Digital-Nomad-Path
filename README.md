# Digital-Nomad-Path
Visualisation of Countries, Donors and Recipients

# Digital Nomad Mobility (Dash)

A standalone Dash web app that visualizes digital nomad mobility flows:
- Select a single **home base** (green node)
- View the **top 20 visited countries** for that home base (blue nodes)
- Optional: show **current countries** (orange nodes)
- Dotted lines connect home → visited, with thickness proportional to the count

## Repo structure

Recommended layout:

.
├── app.py
├── requirements.txt
├── README.md
└── csv/
    └── Nomad_Master_Sheet_COUNTRY_CLEAN_with_ID.csv

> Note: `app.py` must expose `server = app.server` for gunicorn.

## Run locally

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
