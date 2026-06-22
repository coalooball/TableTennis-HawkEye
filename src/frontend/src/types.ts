import { languageLabels } from "./language";

export type Language = keyof typeof languageLabels;
export type LanguageLabels = (typeof languageLabels)[Language];

export type AppState = {
  source: string | null;
  hasCapture: boolean;
  hasFrame: boolean;
  playing: boolean;
  frameIndex: number;
  fps: number;
  videoName: string | null;
  videoFileSizeBytes: number | null;
  videoWidth: number | null;
  videoHeight: number | null;
  videoFrameCount: number | null;
  videoDurationSeconds: number | null;
  poseModelPath: string;
  ballModelPath: string;
  tableModelPath: string;
  showPosePersonBoxes: boolean;
  showPoseSkeleton: boolean;
  showPoseLabels: boolean;
  showTableBoxes: boolean;
  showBallBoxes: boolean;
  action: string;
  detail: string;
  status: string;
};

export type FrameResponse = {
  state: AppState;
  frame: string | null;
};
