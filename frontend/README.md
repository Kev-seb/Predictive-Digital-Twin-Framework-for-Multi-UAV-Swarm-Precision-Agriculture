# Garuda — renovated frontend

A separate Streamlit interface inspired by the layout and navy/amber styling of the Dhinu fitness dashboard. The original `src/dashboard` files and root Streamlit configuration are unchanged.

## Launch

From the project root in PowerShell:

```powershell
.\frontend\run.ps1
```

Open http://localhost:8502. The launcher selects the correct working directory and theme without modifying the original `.streamlit/config.toml`.

The original UI can still be started independently:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/dashboard/dashboard.py --server.port 8501
```

Install the frontend requirements into your project environment with `python -m pip install -r frontend/requirements.txt`. The renovated frontend requires Streamlit 1.58 or newer within major version 1. No JavaScript build or additional packages are required. Run from the repository root if launching `frontend/app.py` manually; model and export paths in the existing workflows are relative to that root.

## Structure

- `app.py`: entrypoint, routing, shared layout, and lazy telemetry startup.
- `navigation.py`: grouped navigation labels, icons, and URLs.
- `ui.py`: sidebar, persistent field settings, workspace bar, headings, and metric cards.
- `assets/theme.css`: design tokens, navigation, controls, cards, and responsive styling.
- `views/overview.py`: redesigned overview with actual survey statistics and clear empty states.
- `views/01_…14_…`: separate copies of the 14 workflows, adapted to the new shell.
- `frontend_shared.py`: local copy of the dashboard support layer; imports the existing scientific modules from `src`.
- `checks/smoke.py`: survey-processing, navigation, state-persistence, and original-file-integrity checks.

`views` deliberately replaces Streamlit's conventional `pages` folder: routing is explicitly owned by `st.navigation`, avoiding automatic page discovery.

## Using the workspace

1. Open **Upload & process**, then upload a Green / Red / Red Edge / NIR GeoTIFF or explicitly enable **Demo Mode**.
2. Use the grouped sidebar to move through analysis, mapping, operations, and intelligence.
3. Expand **Field settings** at the bottom of the sidebar to change the field name, growth stage, coordinates, or thresholds. These settings persist across page changes.
4. On small screens, use the sidebar toggle at the top to open navigation.

Demo data is labeled as synthetic on the overview. Empty metrics are shown as dashes rather than invented measurements. Crop stage is a user-selected setting, not an inferred prediction.

## Isolation and existing integrations

- Both UIs reuse the same scientific modules and model files. This is a separate presentation layer, not a duplicate backend repository.
- The new digital twin writes its history beneath `frontend/runtime/digital_twin_memory` to preserve the original twin history.
- Other existing export workflows retain their existing output destinations.
- HTTP telemetry uses ports 8021–8041 and WebSockets use 8786–8806, separate from the original defaults. Services start when opening the spatial or UAV operations screens.
- MAVLink connection settings still need to match your simulator or vehicle. Avoid connecting both UIs to the same exclusive UDP listener at the same time.
- Weather and Earth Engine continue to require their existing network access/authentication. The Three.js viewer loads its existing CDN dependencies.
- The original scientific models, training limitations, and simulation assumptions remain applicable. Renovating the UI does not validate their predictions.

## Verification

```powershell
.\.venv\Scripts\python.exe frontend/checks/smoke.py
```

The smoke check opens all 14 initial screens, processes the synthetic survey, renders vegetation, zoning, temporal, optimizer, digital twin and report pages, checks that crop-stage settings survive navigation, and verifies hashes of the original dashboard Python files. Desktop and mobile navigation were also inspected in the browser. Live vehicle connectivity, trained-model inference, and external service responses require their respective environments.
