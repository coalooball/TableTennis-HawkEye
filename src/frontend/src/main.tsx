import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { open } from "@tauri-apps/plugin-dialog";
import { App as AntApp, Button, ConfigProvider, Flex, Input, Layout, Modal, Space, Typography, theme } from "antd";
import { MainWorkspace } from "./MainWorkspace";
import { NavigationBar } from "./NavigationBar";
import { defaultLanguage, languageLabels } from "./language";
import "./styles.css";
import type { AppState, FrameResponse, Language } from "./types";

const API_BASE = "http://127.0.0.1:8765";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

function DesktopApp() {
  const { message } = AntApp.useApp();
  const [state, setState] = useState<AppState | null>(null);
  const [frame, setFrame] = useState<string | null>(null);
  const [confidence, setConfidenceValue] = useState(35);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isCalibratingTable, setIsCalibratingTable] = useState(false);
  const [language, setLanguage] = useState<Language>(defaultLanguage);
  const [modelPath, setModelPath] = useState("models/yolo26n-pose.pt");
  const [ballModelPath, setBallModelPath] = useState("models/yolo11s-ball.pt");
  const [tableModelPath, setTableModelPath] = useState("models/table.pt");
  const [saveFrameDirectory, setSaveFrameDirectory] = useState("outputs");
  const timer = useRef<number | null>(null);
  const playingRef = useRef(false);

  const applyState = (next: AppState) => {
    setState(next);
    setModelPath(next.poseModelPath);
    setBallModelPath(next.ballModelPath);
    setTableModelPath(next.tableModelPath);
  };

  const applyFrameResponse = (response: FrameResponse) => {
    applyState(response.state);
    setFrame(response.frame);
  };

  const requestNextFrame = async () => {
    if (!playingRef.current) {
      return;
    }
    try {
      const response = await api<FrameResponse>("/frame/next", { method: "POST" });
      if (!playingRef.current) {
        return;
      }
      applyFrameResponse(response);
      if (!response.state.playing) {
        setPlayback(false);
        return;
      }
    } catch (error) {
      setPlayback(false);
      message.error(String(error));
      return;
    }

    timer.current = window.setTimeout(requestNextFrame, 33);
  };

  const setPlayback = (active: boolean) => {
    playingRef.current = active;
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    if (active) {
      timer.current = window.setTimeout(requestNextFrame, 0);
    }
  };

  useEffect(() => {
    api<AppState>("/state")
      .then(applyState)
      .catch((error) => message.error(String(error)));
    return () => setPlayback(false);
  }, []);

  const openVideo = async () => {
    const path = await open({
      multiple: false,
      filters: [{ name: labels.videos, extensions: ["mp4", "mov", "avi", "mkv", "webm"] }],
    });
    if (!path) {
      return;
    }
    const response = await api<FrameResponse>("/source/video", {
      method: "POST",
      body: JSON.stringify({ path: String(path) }),
    });
    applyFrameResponse(response);
  };

  const pickModel = async (setter: (path: string) => void) => {
    const path = await open({
      multiple: false,
      filters: [{ name: labels.yoloModels, extensions: ["pt", "onnx"] }],
    });
    if (path) {
      setter(String(path));
    }
  };

  const pickDirectory = async (setter: (path: string) => void) => {
    const path = await open({
      multiple: false,
      directory: true,
    });
    if (path) {
      setter(String(path));
    }
  };

  const openCamera = async () => {
    const response = await api<FrameResponse>("/source/camera", { method: "POST" });
    applyFrameResponse(response);
  };

  const start = async () => {
    const next = await api<AppState>("/play", { method: "POST" });
    applyState(next);
    setPlayback(next.playing);
  };

  const stop = async () => {
    setPlayback(false);
    const next = await api<AppState>("/stop", { method: "POST" });
    applyState(next);
  };

  const save = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const directory = saveFrameDirectory.trim() || "outputs";
    const path = `${directory.replace(/[\\/]+$/, "")}/frame_${timestamp}.jpg`;
    const next = await api<AppState>("/frame/save", {
      method: "POST",
      body: JSON.stringify({ path }),
    });
    applyState(next);
    message.success(next.status);
  };

  const canCalibrateTable = () => {
    if (!state?.hasCapture || !state.hasFrame || !state.videoName) {
      Modal.warning({ title: labels.calibrateTable, content: labels.tableNeedSource });
      return false;
    }
    if (state.playing || playingRef.current) {
      Modal.warning({ title: labels.calibrateTable, content: labels.tableNeedStopped });
      return false;
    }
    return true;
  };

  const calibrateManually = async () => {
    if (!canCalibrateTable()) {
      return;
    }
    const response = await api<FrameResponse>("/table/calibrate", { method: "POST" });
    applyFrameResponse(response);
    setIsCalibratingTable(true);
    message.info(labels.calibratingTable);
  };

  const calibrateAutomatically = async () => {
    if (!canCalibrateTable()) {
      return;
    }
    setIsCalibratingTable(false);
    const response = await api<FrameResponse>("/table/auto-calibrate", { method: "POST" });
    applyFrameResponse(response);
    message.info(response.state.status);
  };

  const clear = async () => {
    const response = await api<FrameResponse>("/source/clear", { method: "POST" });
    applyFrameResponse(response);
    setPlayback(false);
    setIsCalibratingTable(false);
  };

  const addTablePoint = async (event: React.MouseEvent<HTMLImageElement>) => {
    const image = event.currentTarget;
    const rect = image.getBoundingClientRect();
    const x = Math.round(((event.clientX - rect.left) / rect.width) * image.naturalWidth);
    const y = Math.round(((event.clientY - rect.top) / rect.height) * image.naturalHeight);
    const response = await api<FrameResponse>("/table/point", {
      method: "POST",
      body: JSON.stringify({ x, y }),
    });
    applyFrameResponse(response);
    const complete = response.state.status === "Table calibrated";
    setIsCalibratingTable(!complete);
    message.info(response.state.status);
  };

  const setConfidence = async (value: number) => {
    setConfidenceValue(value);
    const next = await api<AppState>("/settings/confidence", {
      method: "POST",
      body: JSON.stringify({ confidence: value / 100 }),
    });
    applyState(next);
  };

  const saveSettings = async () => {
    const poseState = await api<AppState>("/settings/model", {
      method: "POST",
      body: JSON.stringify({ path: modelPath }),
    });
    const next = await api<AppState>("/settings/ball-model", {
      method: "POST",
      body: JSON.stringify({ path: ballModelPath }),
    });
    const tableState = await api<AppState>("/settings/table-model", {
      method: "POST",
      body: JSON.stringify({ path: tableModelPath }),
    });
    applyState(tableState);
    setSettingsOpen(false);
    message.success(`${poseState.status}; ${next.status}; ${tableState.status}`);
  };

  const labels = languageLabels[language];

  return (
    <ConfigProvider theme={{ algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
      <Layout className={`shell ${isDarkMode ? "themeDark" : "themeLight"}`}>
        <NavigationBar
          isDarkMode={isDarkMode}
          labels={labels}
          onOpenVideo={openVideo}
          onOpenCamera={openCamera}
          onOpenSettings={() => setSettingsOpen(true)}
          onToggleTheme={() => setIsDarkMode((value) => !value)}
          onToggleLanguage={() => setLanguage((value) => (value === "en" ? "zh" : "en"))}
        />
        <MainWorkspace
          state={state}
          frame={frame}
          confidence={confidence}
          labels={labels}
          isCalibratingTable={isCalibratingTable}
          onStart={start}
          onStop={stop}
          onManualTableCalibration={calibrateManually}
          onAutoTableCalibration={calibrateAutomatically}
          onSave={save}
          onClear={clear}
          onConfidenceChange={setConfidence}
          onFrameClick={addTablePoint}
        />
        <Modal title={labels.settings} open={settingsOpen} onCancel={() => setSettingsOpen(false)} onOk={saveSettings} okText={labels.save}>
          <Flex vertical gap={12}>
            <Flex vertical gap={8}>
              <Typography.Text>{labels.poseModelPath}</Typography.Text>
              <Space.Compact className="modelPathInput">
                <Input value={modelPath} onChange={(event) => setModelPath(event.target.value)} />
                <Button onClick={() => pickModel(setModelPath)}>{labels.browse}</Button>
              </Space.Compact>
            </Flex>
            <Flex vertical gap={8}>
              <Typography.Text>{labels.ballModelPath}</Typography.Text>
              <Space.Compact className="modelPathInput">
                <Input value={ballModelPath} onChange={(event) => setBallModelPath(event.target.value)} />
                <Button onClick={() => pickModel(setBallModelPath)}>{labels.browse}</Button>
              </Space.Compact>
            </Flex>
            <Flex vertical gap={8}>
              <Typography.Text>{labels.tableModelPath}</Typography.Text>
              <Space.Compact className="modelPathInput">
                <Input value={tableModelPath} onChange={(event) => setTableModelPath(event.target.value)} />
                <Button onClick={() => pickModel(setTableModelPath)}>{labels.browse}</Button>
              </Space.Compact>
            </Flex>
            <Flex vertical gap={8}>
              <Typography.Text>{labels.saveFrameDirectory}</Typography.Text>
              <Space.Compact className="modelPathInput">
                <Input value={saveFrameDirectory} onChange={(event) => setSaveFrameDirectory(event.target.value)} />
                <Button onClick={() => pickDirectory(setSaveFrameDirectory)}>{labels.browse}</Button>
              </Space.Compact>
            </Flex>
          </Flex>
        </Modal>
      </Layout>
    </ConfigProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AntApp>
      <DesktopApp />
    </AntApp>
  </React.StrictMode>,
);
