import { Button, Divider, Dropdown, Flex, Slider, Space, Typography } from "antd";
import { Play, Save, ScanSearch, Square, Target, Trash2 } from "lucide-react";

import type { AppState, LanguageLabels } from "./types";

function formatDuration(seconds: number) {
  const totalSeconds = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

type SidebarProps = {
  state: AppState | null;
  confidence: number;
  labels: LanguageLabels;
  onStart: () => void;
  onStop: () => void;
  onManualTableCalibration: () => void;
  onAutoTableCalibration: () => void;
  onSave: () => void;
  onClear: () => void;
  onConfidenceChange: (value: number) => void;
};

export function Sidebar({
  state,
  confidence,
  labels,
  onStart,
  onStop,
  onManualTableCalibration,
  onAutoTableCalibration,
  onSave,
  onClear,
  onConfidenceChange,
}: SidebarProps) {
  const hasCapture = Boolean(state?.hasCapture);
  const hasFrame = Boolean(state?.hasFrame);
  const playing = Boolean(state?.playing);
  const videoFileSize = state?.videoFileSizeBytes != null ? formatBytes(state.videoFileSizeBytes) : null;
  const videoSize = state?.videoWidth && state.videoHeight ? `${state.videoWidth} x ${state.videoHeight}` : null;
  const videoDuration = state?.videoDurationSeconds != null ? formatDuration(state.videoDurationSeconds) : null;
  const videoFps = state?.fps ? state.fps.toFixed(2) : null;

  return (
    <aside className="sidePanel">
      <Space className="controlStack" direction="vertical" size={8}>
        <Button
          block
          icon={playing ? <Square size={16} /> : <Play size={16} />}
          disabled={!hasCapture}
          onClick={playing ? onStop : onStart}
        >
          {playing ? labels.stop : labels.start}
        </Button>
        <Dropdown
          menu={{
            items: [
              { key: "manual", icon: <Target size={16} />, label: labels.tableManualCalibration, onClick: onManualTableCalibration },
              { key: "auto", icon: <ScanSearch size={16} />, label: labels.tableAutoCalibration, onClick: onAutoTableCalibration },
            ],
          }}
          trigger={["click"]}
        >
          <Button block icon={<Target size={16} />} disabled={!hasFrame}>
            {labels.calibrateTable}
          </Button>
        </Dropdown>
        <Button block icon={<Save size={16} />} disabled={!hasFrame} onClick={onSave}>
          {labels.saveFrame}
        </Button>
        <Button block icon={<Trash2 size={16} />} disabled={!hasFrame} onClick={onClear}>
          {labels.clearTrajectory}
        </Button>
      </Space>
      <Divider />
      <Flex vertical gap={8}>
        <Typography.Text>
          {labels.confidence}: {(confidence / 100).toFixed(2)}
        </Typography.Text>
        <Slider min={10} max={90} value={confidence} onChange={onConfidenceChange} />
      </Flex>
      <Divider />
      <Flex vertical gap={4}>
        <Typography.Text type="secondary">{labels.status}</Typography.Text>
        {state?.videoName ? (
          <Typography.Paragraph className="detail">
            {labels.videoName}: {state.videoName}
            {videoFileSize ? `\n${labels.videoFileSize}: ${videoFileSize}` : ""}
            {videoSize ? `\n${labels.videoSize}: ${videoSize}` : ""}
            {videoDuration ? `\n${labels.videoDuration}: ${videoDuration}` : ""}
            {videoFps ? `\n${labels.videoFps}: ${videoFps}` : ""}
            {state.videoFrameCount ? `\n${labels.videoFrames}: ${state.videoFrameCount}` : ""}
          </Typography.Paragraph>
        ) : (
          <Typography.Paragraph className="detail">{state?.status ?? labels.selectSource}</Typography.Paragraph>
        )}
      </Flex>
    </aside>
  );
}
