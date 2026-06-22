from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.backend import (
    AnalyzeResult,
    AppState,
    ConfidenceRequest,
    FrameSeekRequest,
    FrameResponse,
    ModelPathRequest,
    PoseOverlayRequest,
    SaveFrameRequest,
    SourceRequest,
    TablePointRequest,
    TableTennisBackend,
)


backend = TableTennisBackend()
app = FastAPI(title="TableTennis-HawkEye API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "http://127.0.0.1:1420", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/state", response_model=AppState, response_model_by_alias=True)
def get_state() -> AppState:
    with backend.lock:
        return backend.state()


@app.post("/source/video", response_model=FrameResponse, response_model_by_alias=True)
def open_video(request: SourceRequest) -> FrameResponse:
    with backend.lock:
        if request.path is None:
            return backend.frame_response()
        return backend.open_video(request.path)


@app.post("/source/camera", response_model=FrameResponse, response_model_by_alias=True)
def open_camera() -> FrameResponse:
    with backend.lock:
        return backend.open_camera()


@app.post("/source/clear", response_model=FrameResponse, response_model_by_alias=True)
def clear_source() -> FrameResponse:
    with backend.lock:
        return backend.clear_source()


@app.post("/play", response_model=AppState, response_model_by_alias=True)
def start() -> AppState:
    with backend.lock:
        return backend.start()


@app.post("/stop", response_model=AppState, response_model_by_alias=True)
def stop() -> AppState:
    with backend.lock:
        return backend.stop()


@app.post("/frame/next", response_model=FrameResponse, response_model_by_alias=True)
def next_frame() -> FrameResponse:
    with backend.lock:
        return backend.next_frame()


@app.post("/frame/seek", response_model=FrameResponse, response_model_by_alias=True)
def seek_frame(request: FrameSeekRequest) -> FrameResponse:
    with backend.lock:
        return backend.seek_frame(request.frame_index)


@app.post("/frame/infer", response_model=FrameResponse, response_model_by_alias=True)
def infer_current_frame() -> FrameResponse:
    with backend.lock:
        return backend.infer_current_frame()


@app.post("/settings/confidence", response_model=AppState, response_model_by_alias=True)
def set_confidence(request: ConfidenceRequest) -> AppState:
    with backend.lock:
        return backend.set_confidence(request.confidence)


@app.post("/settings/pose-overlay", response_model=FrameResponse, response_model_by_alias=True)
def set_pose_overlay(request: PoseOverlayRequest) -> FrameResponse:
    with backend.lock:
        return backend.set_pose_overlay(
            request.show_person_boxes,
            request.show_skeleton,
            request.show_labels,
            request.show_table_boxes,
            request.show_ball_boxes,
        )


@app.post("/settings/model", response_model=AppState, response_model_by_alias=True)
def set_model_path(request: ModelPathRequest) -> AppState:
    with backend.lock:
        return backend.set_model_path(request.path)


@app.post("/settings/ball-model", response_model=AppState, response_model_by_alias=True)
def set_ball_model_path(request: ModelPathRequest) -> AppState:
    with backend.lock:
        return backend.set_ball_model_path(request.path)


@app.post("/settings/table-model", response_model=AppState, response_model_by_alias=True)
def set_table_model_path(request: ModelPathRequest) -> AppState:
    with backend.lock:
        return backend.set_table_model_path(request.path)


@app.post("/frame/save", response_model=AppState, response_model_by_alias=True)
def save_current_frame(request: SaveFrameRequest) -> AppState:
    with backend.lock:
        return backend.save_current_frame(request.path)


@app.post("/table/calibrate", response_model=FrameResponse, response_model_by_alias=True)
def calibrate_table() -> FrameResponse:
    with backend.lock:
        return backend.calibrate_table()


@app.post("/table/auto-calibrate", response_model=FrameResponse, response_model_by_alias=True)
def auto_calibrate_table() -> FrameResponse:
    with backend.lock:
        return backend.auto_calibrate_table()


@app.post("/table/point", response_model=FrameResponse, response_model_by_alias=True)
def add_table_point(request: TablePointRequest) -> FrameResponse:
    with backend.lock:
        return backend.add_table_point(request.x, request.y)


@app.post("/trajectory/clear", response_model=AppState, response_model_by_alias=True)
def clear_tracking() -> AppState:
    with backend.lock:
        return backend.clear_tracking()


@app.post("/video/analyze", response_model=AnalyzeResult, response_model_by_alias=True)
def analyze_current_video() -> AnalyzeResult:
    with backend.lock:
        try:
            return backend.analyze_current_video()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
