import { useEffect, useState } from "react";
import { Layout, Slider, Typography } from "antd";

import { Sidebar } from "./Sidebar";
import type { AppState, LanguageLabels } from "./types";

type MainWorkspaceProps = {
  state: AppState | null;
  frame: string | null;
  confidence: number;
  labels: LanguageLabels;
  isCalibratingTable: boolean;
  onStart: () => void;
  onStop: () => void;
  onManualTableCalibration: () => void;
  onAutoTableCalibration: () => void;
  onSave: () => void;
  onClear: () => void;
  onConfidenceChange: (value: number) => void;
  onPoseOverlayChange: (next: {
    showPersonBoxes: boolean;
    showSkeleton: boolean;
    showLabels: boolean;
    showTableBoxes: boolean;
    showBallBoxes: boolean;
  }) => void;
  onSeekFrame: (frameIndex: number) => void;
  onFrameClick: (event: React.MouseEvent<HTMLImageElement>) => void;
};

export function MainWorkspace({
  state,
  frame,
  confidence,
  labels,
  isCalibratingTable,
  onStart,
  onStop,
  onManualTableCalibration,
  onAutoTableCalibration,
  onSave,
  onClear,
  onConfidenceChange,
  onPoseOverlayChange,
  onSeekFrame,
  onFrameClick,
}: MainWorkspaceProps) {
  const hasVideoProgress = Boolean(state?.videoFrameCount && state.videoFrameCount > 0);
  const totalFrames = state?.videoFrameCount ?? 0;
  const currentFrame = Math.min(state?.frameIndex ?? 0, totalFrames);
  const [sliderFrame, setSliderFrame] = useState(currentFrame);

  useEffect(() => {
    setSliderFrame(currentFrame);
  }, [currentFrame]);

  return (
    <Layout.Content className="workspace">
      <Sidebar
        state={state}
        confidence={confidence}
        labels={labels}
        onStart={onStart}
        onStop={onStop}
        onManualTableCalibration={onManualTableCalibration}
        onAutoTableCalibration={onAutoTableCalibration}
        onSave={onSave}
        onClear={onClear}
        onConfidenceChange={onConfidenceChange}
        onPoseOverlayChange={onPoseOverlayChange}
      />
      <section className="viewerPanel">
        <div className={`viewer ${isCalibratingTable ? "isCalibrating" : ""}`}>
          {frame ? <img src={frame} alt="" onClick={isCalibratingTable ? onFrameClick : undefined} /> : <span>{labels.selectSource}</span>}
        </div>
        {hasVideoProgress ? (
          <div className="videoProgress">
            <Slider
              min={1}
              max={totalFrames}
              value={sliderFrame}
              tooltip={{ formatter: null }}
              onChange={setSliderFrame}
              onChangeComplete={onSeekFrame}
            />
            <Typography.Text type="secondary">
              {sliderFrame} / {totalFrames}
            </Typography.Text>
          </div>
        ) : null}
      </section>
    </Layout.Content>
  );
}
