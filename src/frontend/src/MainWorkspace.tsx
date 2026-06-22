import { Layout } from "antd";

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
  onSave: () => void;
  onCalibrate: () => void;
  onClear: () => void;
  onConfidenceChange: (value: number) => void;
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
  onSave,
  onCalibrate,
  onClear,
  onConfidenceChange,
  onFrameClick,
}: MainWorkspaceProps) {
  return (
    <Layout.Content className="workspace">
      <Sidebar
        state={state}
        confidence={confidence}
        labels={labels}
        onStart={onStart}
        onStop={onStop}
        onSave={onSave}
        onCalibrate={onCalibrate}
        onClear={onClear}
        onConfidenceChange={onConfidenceChange}
      />
      <section className={`viewer ${isCalibratingTable ? "isCalibrating" : ""}`}>
        {frame ? <img src={frame} alt="" onClick={isCalibratingTable ? onFrameClick : undefined} /> : <span>{labels.selectSource}</span>}
      </section>
    </Layout.Content>
  );
}
