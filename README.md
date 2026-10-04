# Predictive Digital Twin Framework for Multi-UAV Swarm Precision Agriculture
## Spatiotemporal Biophysical Modeling · Deep Representation Learning · Reinforcement Learning Optimization · Real-Time Ground Control

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![API](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Build & Tests](https://img.shields.io/badge/tests-passed-brightgreen.svg)](#6-test-suite)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![UE5](https://img.shields.io/badge/Unreal%20Engine-5.x-blueviolet.svg)](#55-unreal-engine-5--unity-digital-twin-bridge)

---

## Abstract

This repository contains the complete source code for an enterprise-grade, research-calibrated **Predictive Digital Twin Framework** for autonomous multi-UAV precision agriculture. The platform operates end-to-end: from raw multispectral GeoTIFF ingestion through AI-driven crop stress analysis, real-time swarm dispatch, live ground-station telemetry, and a persistent Unreal Engine 5 / Unity digital twin that mirrors field biophysics in 3D.

**Core capabilities at a glance:**

| Layer | Technology |
|---|---|
| Data ingestion | Rasterio · 4-band GeoTIFF (G/R/RE/NIR) |
| Vegetation indices | NDVI · NDRE · NDWI · SAVI · EVI · MSAVI2 · CIRE |
| Semantic segmentation | PyTorch DeepLabV3+ with Grad-CAM XAI |
| Disease spread | Fisher-Kolmogorov PDE · Directed GNN |
| Treatment optimization | Tabular Q-Learning MDP · Monte Carlo VaR/CVaR |
| Swarm coordination | Potential-field path planning · Serpentine sweep · Stress waypoints |
| Live telemetry | MAVLink · WebRTC video · WebSocket streaming |
| Digital twin | FastAPI backend · UE5 / Unity C# bridge · 20 × 20 HISM grid |
| Mobile | Progressive Web App · WebRTC · offline sensor fusion |
| Research pipeline | 7-stage reproducible ML pipeline · spectral ablation |
| Dashboard | Streamlit multi-tab · dark/light mode · interactive maps |

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Module Reference](#2-module-reference)
3. [Theoretical Formulations](#3-theoretical-formulations)
4. [Research Pipeline](#4-research-pipeline)
5. [Installation & Run Guide](#5-installation--run-guide)
6. [Test Suite](#6-test-suite)
7. [Feature Matrix](#7-feature-matrix)
8. [Academic Documentation](#8-academic-documentation)

---

## 1. System Architecture

```
uav-crop-stress-intelligence/
│
├── .streamlit/                        # Streamlit theme config (dark/light)
│
├── data/
│   ├── raw/                           # Original multispectral GeoTIFF flights
│   ├── processed/                     # Masked maps, zone boundaries, twin state
│   ├── manifests/                     # Flight manifest, patch manifest CSV (5 487 patches)
│   │   ├── flight_manifest.csv
│   │   ├── patch_manifest.csv
│   │   ├── patch_summary.json
│   │   └── dataset_statistics.json
│   └── splits/
│       └── splits_manifest.json       # Flight-disjoint train/val/test splits
│
├── models/
│   ├── segmentation/                  # DeepLabV3+ weights (.pth)
│   └── classifiers/                   # EfficientNet-B0 weights (.pt)
│
├── research/                          # 7-stage reproducible ML pipeline
│   ├── 01_build_data_manifest.py
│   ├── 02_extract_patches_with_provenance.py
│   ├── 03_create_flight_disjoint_splits.py
│   ├── 04_train_spectral_ablation.py
│   ├── 05_stress_proxy_generator.py
│   ├── 06_radiometric_normalization.py
│   ├── 07_train_stress_classifier.py
│   └── reproducibility/
│       └── capture_environment.py
│
├── scripts/                           # Utility & validation scripts
│   ├── benchmark_optimizer.py         # RL optimizer benchmarking (33 KB)
│   ├── sensitivity_analysis.py        # Parameter sensitivity analysis
│   ├── simulate_unity_client.py       # HTTP API simulation client
│   ├── validate_digital_twin.py       # End-to-end digital twin validation
│   ├── validate_ndvi.py               # NDVI reference validation
│   ├── verify_sitl_telemetry.py       # Software-in-the-loop telemetry check
│   ├── batch_processing.py            # Batch GeoTIFF processing
│   ├── generate_sample_tiff.py        # Synthetic test data generator
│   └── prepare_dataset.py             # Dataset preparation utilities
│
├── src/                               # Core engine (24 modules)
│   ├── ai_engine/                     # Treatment optimization & yield
│   │   ├── treatment_optimizer.py     # Q-Learning MDP, Knapsack, Monte Carlo
│   │   ├── treatment_recommender.py   # Rule-based prescription recommender
│   │   ├── yield_predictor.py         # Monteith LUE biomass & GDD yield model
│   │   ├── disease_evolution.py       # SIR/SEIR disease evolution model
│   │   └── epidemiology.py            # Spatiotemporal epidemiology engine
│   │
│   ├── classification/
│   │   └── train_classifier.py        # EfficientNet-B0 two-phase transfer learning
│   │
│   ├── config/
│   │   └── config.py                  # Pydantic BaseSettings environment config
│   │
│   ├── core/
│   │   └── image_loader.py            # Rasterio GeoTIFF wrappers & band alignment
│   │
│   ├── dashboard/
│   │   └── dashboard.py               # Streamlit multi-tab UI (all features)
│   │
│   ├── database.py                    # SQLite mission & telemetry database layer
│   │
│   ├── digital_twin/                  # Spatiotemporal simulation engines
│   │   ├── twin.py                    # Digital twin state manager & UE5 sync
│   │   ├── simulator.py               # Field physics & crop growth simulator
│   │   ├── flight_physics.py          # UAV kinematics & potential-field planner
│   │   ├── gpu_physics.py             # PyTorch GPU particle spray simulation
│   │   └── camera_feed.py             # Simulated UAV camera feed generator
│   │
│   ├── gis/
│   │   └── mapping.py                 # Folium satellite basemap & zone overlay
│   │
│   ├── ground_station/                # Real-time ground control station
│   │   ├── live_field_tab.py          # Live field monitoring dashboard tab
│   │   ├── live_telemetry_panel.py    # MAVLink telemetry display panel
│   │   ├── live_ai_panel.py           # Real-time AI inference results panel
│   │   ├── live_video_panel.py        # WebRTC live video feed panel
│   │   ├── live_qos_panel.py          # Link quality & QoS metrics panel
│   │   ├── live_report_generator.py   # Auto-generated mission reports
│   │   ├── mission_map_panel.py       # Interactive mission map with waypoints
│   │   └── mission_replay_panel.py    # Recorded mission replay & analysis
│   │
│   ├── indices/                       # Vectorized remote sensing indices
│   │   ├── indices.py                 # Central multispectral calculation router
│   │   ├── ndvi.py                    # Normalized Difference Vegetation Index
│   │   ├── ndre.py                    # Normalized Difference Red Edge Index
│   │   ├── ndwi.py                    # Normalized Difference Water Index
│   │   ├── savi.py                    # Soil-Adjusted Vegetation Index
│   │   ├── evi.py                     # Enhanced Vegetation Index
│   │   ├── msavi2.py                  # Modified SAVI 2
│   │   ├── cire.py                    # Chlorophyll Index Red Edge
│   │   └── stress_score.py            # Multi-index weighted composite stress
│   │
│   ├── live_mode/                     # Real-time live field operations
│   │   ├── live_field_controller.py   # Live field state machine & dispatch
│   │   └── alert_engine.py            # Threshold-based alert & notification engine
│   │
│   ├── mission/                       # Mission lifecycle management
│   │   ├── mission_object.py          # Mission data model & waypoint definitions
│   │   ├── mission_recorder.py        # Flight-to-disk mission recorder
│   │   └── mission_database.py        # SQLite-backed mission persistence
│   │
│   ├── mobile/
│   │   └── companion_app/             # Progressive Web App (PWA)
│   │       ├── index.html             # App shell & UI layout
│   │       ├── app.js                 # Core app logic & API integration
│   │       ├── sensor_manager.js      # GPS/IMU/compass sensor access
│   │       ├── webrtc_client.js       # WebRTC video streaming client
│   │       ├── calibration.js         # Sensor calibration routines
│   │       ├── offline_manager.js     # IndexedDB offline data storage
│   │       ├── service_worker.js      # PWA service worker & cache
│   │       └── manifest.json          # Web app manifest
│   │
│   ├── mobile_api/
│   │   └── mobile_api_server.py       # FastAPI server for companion app
│   │
│   ├── reports/
│   │   └── pdf_report.py              # FPDF2 offline PDF report compiler
│   │
│   ├── rgb_ai/                        # RGB-only AI inference engine
│   │   ├── rgb_inference_engine.py    # Unified RGB inference coordinator
│   │   ├── rgb_crop_stage_classifier.py  # Growth stage classification
│   │   ├── rgb_disease_detector.py    # Disease symptom detection
│   │   ├── rgb_stress_estimator.py    # Visual stress intensity estimation
│   │   └── rgb_weed_detector.py       # Weed presence detection
│   │
│   ├── segmentation/                  # DeepLabV3+ segmentation pipeline
│   │   ├── deeplabv3_model.py         # PyTorch DeepLabV3+ custom backbone
│   │   ├── gradcam_segmentation.py    # Grad-CAM explainability saliency
│   │   ├── stress_segmentation.py     # Full stress segmentation pipeline
│   │   ├── predict_segmentation.py    # Inference-time prediction wrapper
│   │   ├── segmentation_metrics.py    # mIoU, F1, confusion matrix
│   │   └── train_segmentation.py      # Training & validation loop
│   │
│   ├── spatial/
│   │   └── reconstruction.py          # ORB stitching · DSM · Canopy CHM
│   │
│   ├── streaming/                     # WebRTC video streaming
│   │   ├── webrtc_bridge.py           # aiortc signalling & STUN/TURN bridge
│   │   ├── frame_interface.py         # Camera frame abstraction layer
│   │   └── frame_queue.py             # Lock-free frame queue for low latency
│   │
│   ├── telemetry/                     # Sensor telemetry processing
│   │   ├── phone_telemetry_adapter.py # Smartphone GPS/IMU → MAVLink adapter
│   │   └── sensor_fusion.py           # Extended Kalman Filter sensor fusion
│   │
│   ├── temporal/                      # Change detection & growth tracking
│   │   ├── change_detection.py        # Z-score differencing & CVA analysis
│   │   └── growth_stage_tracking.py   # GDD-based phenology stage tracker
│   │
│   ├── unity_integration/             # Game engine digital twin bridge
│   │   ├── UnityBridgeClient.cs       # C# Unity/UE5 HTTP API client
│   │   └── README.md                  # Integration setup guide
│   │
│   ├── vision/                        # Low-level computer vision
│   │   ├── camera_calibrator.py       # Intrinsic/extrinsic camera calibration
│   │   └── image_quality_assessor.py  # BRISQUE/NIQE quality assessment
│   │
│   └── weather/
│       ├── openmeteo_client.py        # Open-Meteo REST weather client
│       └── weather_risk_engine.py     # Spray feasibility window assessment
│
├── tests/                             # Pytest regression suite (8 modules)
│
├── digital_twin_api.py                # FastAPI UE5 backend (20×20 HISM grid)
├── academic_paper.tex                 # Full IEEE-format LaTeX paper
├── academic_paper_draft.md            # Markdown paper draft
├── academic_paper_outline.md          # Structured paper outline
└── requirements.txt                   # All dependencies
```

---

## 2. Module Reference

### 2.1 `src/ai_engine/` — Treatment Optimization & Prediction

| File | Description |
|---|---|
| `treatment_optimizer.py` | Tabular Q-Learning MDP (ε-greedy), Knapsack multi-resource scheduling, Monte Carlo rollouts, VaR/CVaR risk estimation. **36 KB** |
| `treatment_recommender.py` | Rule-based prescription engine; outputs variable-rate fertilizer, fungicide, and irrigation recommendations per zone |
| `yield_predictor.py` | Monteith Light Use Efficiency biomass model + Growing Degree Days (GDD) thermal accumulation + heat-sterility harvest index |
| `disease_evolution.py` | SIR/SEIR compartmental disease state evolution with weather coupling |
| `epidemiology.py` | Anisotropic Fisher-Kolmogorov PDE spread + Directed GNN transmission routing |

### 2.2 `src/digital_twin/` — Spatiotemporal Simulation

| File | Description |
|---|---|
| `twin.py` | Persistent digital twin state manager; synchronises field JSON state to UE5 via REST |
| `simulator.py` | Full field physics simulator: crop growth, stress propagation, nutrient cycles |
| `flight_physics.py` | UAV 6-DoF kinematics, potential-field collision avoidance, serpentine sweep, stress-proportional waypoint generation |
| `gpu_physics.py` | PyTorch GPU particle spray engine: droplet advection, wind drift, gravity, turbulence |
| `camera_feed.py` | Synthetic simulated UAV camera feed with configurable noise and altitude |

### 2.3 `src/ground_station/` — Real-Time Ground Control Station

The ground station provides a full operational interface for live UAV missions:

- **`live_field_tab.py`** — Live field monitoring with real-time NDVI heatmap refresh
- **`live_telemetry_panel.py`** — MAVLink telemetry display (battery, altitude, speed, GPS)
- **`live_ai_panel.py`** — Real-time AI inference results streamed from edge UAV
- **`live_video_panel.py`** — WebRTC live video feed from onboard camera
- **`live_qos_panel.py`** — Link quality, latency, packet loss, and RSSI indicators
- **`live_report_generator.py`** — Auto-generated mission PDF reports with flight logs
- **`mission_map_panel.py`** — Interactive Folium map with live UAV position and waypoints
- **`mission_replay_panel.py`** — Post-flight mission replay with timeline scrubbing

### 2.4 `src/live_mode/` — Real-Time Field Operations

- **`live_field_controller.py`** — State machine managing live UAV dispatch, field update cycles, and swarm coordination triggers
- **`alert_engine.py`** — Threshold-based alert system: stress exceedance, battery low, geofence breach, link loss

### 2.5 `src/mission/` — Mission Lifecycle

- **`mission_object.py`** — Mission data model: waypoints, spray parameters, crop zones, flight plan
- **`mission_recorder.py`** — Records flight telemetry, AI inferences, and images to disk
- **`mission_database.py`** — SQLite-backed mission persistence, query, and replay

### 2.6 `src/mobile/companion_app/` — Progressive Web App

A fully offline-capable smartphone companion app:

| File | Purpose |
|---|---|
| `index.html` | App shell, responsive layout, tab navigation |
| `app.js` | Core logic, API integration, real-time data polling |
| `sensor_manager.js` | Accesses device GPS, accelerometer, gyroscope, compass |
| `webrtc_client.js` | WebRTC peer connection for live UAV video streaming |
| `calibration.js` | In-app IMU and compass calibration routines |
| `offline_manager.js` | IndexedDB caching for offline field data access |
| `service_worker.js` | PWA service worker; enables offline usage and background sync |
| `manifest.json` | Web app manifest for installable PWA |

### 2.7 `src/rgb_ai/` — RGB-Only AI Inference

Enables AI analysis even without multispectral data (standard camera):

- **`rgb_inference_engine.py`** — Unified coordinator for all RGB models
- **`rgb_crop_stage_classifier.py`** — Growth stage classification (seedling → mature)
- **`rgb_disease_detector.py`** — Visual disease symptom detection
- **`rgb_stress_estimator.py`** — Canopy stress estimation from colour features
- **`rgb_weed_detector.py`** — Weed detection for precision herbicide application

### 2.8 `src/segmentation/` — DeepLabV3+ Semantic Segmentation

| File | Description |
|---|---|
| `deeplabv3_model.py` | PyTorch DeepLabV3+ with 4-channel multispectral input backbone |
| `gradcam_segmentation.py` | Grad-CAM spatial attribution saliency maps |
| `stress_segmentation.py` | End-to-end stress segmentation pipeline |
| `predict_segmentation.py` | Inference wrapper with softmax class probability maps |
| `segmentation_metrics.py` | mIoU, class F1, confusion matrix computation |
| `train_segmentation.py` | Training loop with validation, checkpointing, LR scheduling |

### 2.9 `src/streaming/` — WebRTC Video Streaming

- **`webrtc_bridge.py`** — aiortc-based signalling server with STUN/TURN support
- **`frame_interface.py`** — Abstraction layer for GStreamer / OpenCV camera sources
- **`frame_queue.py`** — Lock-free frame queue for sub-100 ms end-to-end latency

### 2.10 `src/telemetry/` — Sensor Telemetry

- **`phone_telemetry_adapter.py`** — Converts smartphone GPS/IMU data to MAVLink messages
- **`sensor_fusion.py`** — Extended Kalman Filter fusing GPS, accelerometer, and barometer

### 2.11 `src/unity_integration/` — UE5 / Unity Digital Twin Bridge

- **`UnityBridgeClient.cs`** — C# Unity/UE5 HTTP client that polls `digital_twin_api.py` and updates Hierarchical Instanced Static Mesh (HISM) material parameters
- **`digital_twin_api.py`** — FastAPI backend serving the 20 × 20 paddy field state grid with endpoints for per-instance updates, anomaly detection, NDVI heatmap, and full state dump

### 2.12 `src/indices/` — Remote Sensing Vegetation Indices

All indices are vectorised with NumPy for zero-copy band operations:

| Index | Formula | Interpretation |
|---|---|---|
| NDVI | $(NIR-R)/(NIR+R)$ | Chlorophyll density & canopy vigour |
| NDRE | $(NIR-RE)/(NIR+RE)$ | Leaf nitrogen content (deep canopy) |
| NDWI | $(G-NIR)/(G+NIR)$ | Canopy water content & waterlogging |
| SAVI | $(NIR-R)(1+L)/(NIR+R+L)$ | Soil-corrected vegetation (L=0.5) |
| EVI | $G(NIR-R)/(NIR+C_1R-C_2B+L)$ | Aerosol-corrected; no saturation |
| MSAVI2 | $(2NIR+1-\sqrt{(2NIR+1)^2-8(NIR-R)})/2$ | Improved soil adjustment |
| CIRE | $NIR/RE-1$ | Canopy chlorophyll content |

---

## 3. Theoretical Formulations

### 3.1 Spatiotemporal Disease Spread

Anisotropic advection-diffusion-reaction PDE:

$$\frac{\partial S}{\partial t} = \nabla \cdot (\mathbf{D} \nabla S) + rS\left(1 - \frac{S}{K}\right) - \vec{v}_{\text{wind}} \cdot \nabla S$$

- $\mathbf{D}$ = diffusion tensor, skewed along wind vector
- $r$ = growth rate dynamically coupled to relative humidity and temperature
- $K$ = NDVI-limited carrying capacity

Directed GNN edge weights:

$$e_{ij} = \text{softmax}\!\left(\text{LeakyReLU}\!\left(\vec{W}^T[\vec{h}_i \| \vec{h}_j] + \theta\cos(\phi_{\text{wind}} - \phi_{ij})\right)\right)$$

### 3.2 Q-Learning Treatment Optimization

$$Q(s, a) \leftarrow Q(s, a) + \alpha\!\left[R(s,a) + \gamma\max_{a'}Q(s',a') - Q(s,a)\right]$$

Reward function (yield vs. cost vs. stress penalty):

$$R(s,a) = \text{Yield}(s') \cdot P_{\text{crop}} - \sum_i \text{Input}_i \cdot C_{\text{chem}} - \lambda \cdot \text{Stress}(s')$$

Monte Carlo risk estimation: $N = 1000$ rollouts for VaR and CVaR at 95th percentile.

### 3.3 Multi-UAV Swarm Path Planning

Potential-field repulsive force between drone $i$ and $j$:

$$\vec{F}_{\text{repulsive},i} = \sum_{j \neq i} \eta\!\left(\frac{1}{d_{ij}} - \frac{1}{d_0}\right)\frac{1}{d_{ij}^2}\hat{u}_{ji}$$

**Stress-proportional spray dosing**: spray rate $\rho_k$ at waypoint $k$ scales with local stress score $\sigma_k$:

$$\rho_k = \rho_{\min} + (\rho_{\max} - \rho_{\min}) \cdot \sigma_k$$

**Serpentine sweep**: boustrophedon path minimises re-crossing and total flight distance.

### 3.4 GPU Particle Spray Engine

Droplet kinematic update (PyTorch CUDA):

$$\vec{x}_p(t+\Delta t) = \vec{x}_p(t) + \left(\vec{v}_{\text{drone}} + \vec{w}_{\text{wind}} + \vec{v}_{\text{turbulent}}\right)\Delta t - \tfrac{1}{2}\vec{g}\Delta t^2$$

### 3.5 Grad-CAM Explainability

Channel importance weight from gradient global average pool:

$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{i,j}}$$

Saliency map:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

### 3.6 Biomass & Yield Forecasting

Monteith Light Use Efficiency model:

$$\text{Biomass} = \sum \left(PAR \times fPAR \times LUE_{\max} \times f(T) \times f(W)\right)$$

Growing Degree Days and heat-stress adjusted harvest index:

$$GDD = \sum_{\text{day}}\!\left(\frac{T_{\max}+T_{\min}}{2} - T_{\text{base}}\right)$$

$$\text{HI} = \text{HI}_{\text{base}} \times \left(1 - \kappa_{\text{heat}} \cdot \text{Days}_{>38^\circ\text{C}}\right), \qquad \text{Yield} = \text{Biomass} \times \text{HI}$$

### 3.7 Sensor Fusion (Extended Kalman Filter)

State vector $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \phi, \theta, \psi]^T$ fused from GPS, barometer, and 6-DoF IMU with process noise $\mathbf{Q}$ and measurement noise $\mathbf{R}$ tuned per sensor.

---

## 4. Research Pipeline

A 7-stage reproducible ML pipeline for crop stress classification research:

| Step | Script | Description |
|---|---|---|
| 01 | `01_build_data_manifest.py` | Scans all UAV flight directories, parses FLIGHT_RE regex, builds flight manifest CSV (20 flights, 8 244 image sets) |
| 02 | `02_extract_patches_with_provenance.py` | Disk-efficient 224 × 224 patch extraction; max 250 patches/flight; float16 `.npy`; ~1.6 GB total |
| 03 | `03_create_flight_disjoint_splits.py` | Creates 4 split types: temporal holdout, cross-field, LOFO, random patch |
| 04 | `04_train_spectral_ablation.py` | Multi-seed (5 seeds) spectral ablation: RGB vs 3-band vs 4-band vs 5-channel; leakage quantification |
| 05 | `05_stress_proxy_generator.py` | NDVI-threshold proxy label generation for weakly-supervised learning |
| 06 | `06_radiometric_normalization.py` | Empirical line calibration + histogram matching across flights |
| 07 | `07_train_stress_classifier.py` | EfficientNet-B0 multi-task classifier (crop stage + stress severity); flight-disjoint validation |

**Reproducibility**: `research/reproducibility/capture_environment.py` snapshots the full conda/pip environment, git commit hash, and hardware specs.

**Dataset**: 20 UAV flights across 2 paddy fields (Field001 3 acres, Field002 2 acres), 5 487 patches total, 4-band multispectral (G/R/RE/NIR).

---

## 5. Installation & Run Guide

### Prerequisites

- Python 3.11 / 3.12 / 3.13
- Windows, Linux, or macOS
- CUDA GPU optional (CPU fallback supported everywhere)

### 5.1 Install

```bash
git clone https://github.com/Kev-seb/Predictive-Digital-Twin-Framework-for-Multi-UAV-Swarm-Precision-Agriculture.git
cd Predictive-Digital-Twin-Framework-for-Multi-UAV-Swarm-Precision-Agriculture

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 5.2 Run the Streamlit Dashboard

```bash
streamlit run src/dashboard/dashboard.py
```

Open **http://localhost:8501** in your browser.

**Dashboard tabs:**
- 🌱 **Vegetation Analytics** — NDVI/NDRE/EVI maps, zone statistics
- 🔬 **Segmentation & XAI** — DeepLabV3+ stress segmentation + Grad-CAM
- 🦠 **Disease Forecasting** — PDE spread maps & GNN transmission graph
- 💊 **Treatment Optimizer** — Q-Learning prescription + Knapsack scheduler
- 🚁 **Swarm Operations** — Multi-UAV path planning & spray dispatch
- 📊 **Yield Forecast** — Biomass model + GDD growth stage tracker
- 🌦️ **Weather Risk** — Spray window feasibility from Open-Meteo API
- 🗺️ **GIS Map** — Satellite basemap with zone overlays
- 🏭 **Ground Station** — Live telemetry, AI results, video, QoS panels
- 📡 **Live Mode** — Real-time field state, alert engine, swarm dispatch
- 📄 **PDF Report** — Offline mission report compiler

### 5.3 Run the Digital Twin API (for UE5 / Unity)

```bash
python digital_twin_api.py
```

API available at **http://127.0.0.1:8008/docs** (Swagger UI).

Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/field/update` | Receive per-instance state from UE5 |
| `GET` | `/field/status` | Full 400-instance field state |
| `GET` | `/field/heatmap` | NDVI grid as 20 × 20 list |
| `GET` | `/field/anomalies` | Instances with NDVI drop above threshold |
| `GET` | `/field/summary` | Aggregated field statistics |
| `DELETE` | `/field/reset` | Reset all instances to defaults |

### 5.4 Run the Mobile API (companion app backend)

```bash
python -m uvicorn src.mobile_api.mobile_api_server:app --host 0.0.0.0 --port 8080
```

Scan the QR code from the dashboard to open the companion app on your phone.

### 5.5 Unreal Engine 5 / Unity Digital Twin Bridge

1. Start `digital_twin_api.py` on the same machine as UE5.
2. In UE5, add `src/unity_integration/UnityBridgeClient.cs` (or the Blueprint equivalent) to your level Actor.
3. Set `ApiBaseUrl = "http://127.0.0.1:8008"` in the component.
4. The C# client polls `/field/status` every 500 ms and updates the HISM material parameters (`NDVI`, `StressLevel`, `GrowthStage`) per-instance on the GPU.

### 5.6 Research Pipeline

Run stages in order from `uav-crop-stress-intelligence/`:

```bash
python research/01_build_data_manifest.py
python research/02_extract_patches_with_provenance.py
python research/03_create_flight_disjoint_splits.py
python research/04_train_spectral_ablation.py
python research/05_stress_proxy_generator.py
python research/06_radiometric_normalization.py
python research/07_train_stress_classifier.py
```

### 5.7 Training Classifiers

Train crop stress classification head on custom drone data:

```bash
python src/classification/train_classifier.py \
    --task stage \
    --data_dir data/processed/classification \
    --epochs 50 \
    --batch_size 16
```

---

## 6. Test Suite

```bash
.venv\Scripts\pytest tests/ -v
```

| Test File | What It Validates |
|---|---|
| `test_ai_optimizer.py` | Q-Learning policy converges; net reward exceeds baseline |
| `test_flight_physics.py` | Potential-field forces maintain $d_{ij} > d_0 = 6\,\text{m}$ across trajectories |
| `test_indices.py` | NDVI, SAVI, NDWI outputs match reference arrays |
| `test_segmentation.py` | DeepLabV3+ forward pass produces correct output shape |
| `test_swarm_coordination.py` | Swarm reaches all waypoints without collision |
| `test_temporal.py` | Change detection outputs plausible difference maps |
| `test_weather.py` | Weather risk engine returns valid spray window |
| `test_yield.py` | Biomass model outputs within $[0.0, 12.0]\,\text{t/ha}$ |

---

## 7. Feature Matrix

| Component | Technology | Status |
|---|---|---|
| **Multispectral Ingestion** | Rasterio · NumPy (4-band GeoTIFF) | ✅ Verified |
| **Vegetation Indices** | NDVI · NDRE · NDWI · SAVI · EVI · MSAVI2 · CIRE | ✅ Verified |
| **Semantic Segmentation** | PyTorch DeepLabV3+ (4-channel backbone) | ✅ Verified |
| **Explainable AI (XAI)** | Grad-CAM pixel-level saliency | ✅ Verified |
| **Photogrammetry** | ORB+RANSAC stitching · Stereo DSM/CHM | ✅ Complete |
| **Disease Forecasting** | Fisher-Kolmogorov PDE · Directed GNN | ✅ Active |
| **Treatment Optimizer** | Q-Learning MDP · Knapsack · Monte Carlo VaR | ✅ Verified |
| **Yield Forecasting** | Monteith LUE · GDD · Heat-sterility index | ✅ Verified |
| **Swarm Path Planning** | Potential-field · Serpentine sweep · Stress waypoints | ✅ Verified |
| **Stress-Proportional Dosing** | Per-waypoint dosing ∝ local stress score | ✅ Active |
| **GPU Spray Physics** | PyTorch CUDA particle drift simulation | ✅ Active |
| **GIS Mapping** | Folium satellite basemap · K-Means zone clustering | ✅ Complete |
| **Weather Risk** | Open-Meteo REST · Spray feasibility windows | ✅ Active |
| **Ground Control Station** | MAVLink telemetry · QoS · mission map/replay | ✅ Complete |
| **Live Mode** | Real-time field state machine · alert engine | ✅ Active |
| **WebRTC Video Streaming** | aiortc · STUN/TURN · lock-free frame queue | ✅ Complete |
| **Sensor Fusion** | Extended Kalman Filter (GPS+IMU+baro) | ✅ Active |
| **RGB-Only AI** | Stage classification · disease · stress · weed | ✅ Complete |
| **Mobile PWA** | WebRTC · offline IndexedDB · service worker | ✅ Complete |
| **UE5 / Unity Bridge** | FastAPI · C# HISM material update client | ✅ Active |
| **Digital Twin API** | 20×20 NDVI grid · anomaly detection · heatmap | ✅ Active |
| **PDF Reports** | FPDF2 offline deterministic report compiler | ✅ Complete |
| **Research Pipeline** | 7-stage reproducible ML pipeline | ✅ Complete |
| **Test Suite** | 8 Pytest modules — all passing | ✅ Verified |
| **Theme Sync** | Auto dark/light mode on UI + matplotlib | ✅ Complete |

---

## 8. Academic Documentation

This project is documented at research-paper level:

- **`academic_paper.tex`** — Full IEEE double-column LaTeX paper covering all theoretical formulations, system architecture, experiments, and results
- **`academic_paper_draft.md`** — Markdown draft for collaborative editing
- **`academic_paper_outline.md`** — Structured section-by-section paper outline

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

---

*Built with ❤️ for sustainable precision agriculture — reducing chemical inputs, maximising crop yield, and enabling autonomous multi-UAV field intelligence.*
