import { Button, Dropdown, Layout, Space } from "antd";
import { Camera, FileVideo, FolderOpen, Languages, Moon, ScanSearch, Settings, Sun, Target } from "lucide-react";

import type { LanguageLabels } from "./types";

type NavigationBarProps = {
  isDarkMode: boolean;
  labels: LanguageLabels;
  onOpenVideo: () => void;
  onOpenCamera: () => void;
  onManualTableCalibration: () => void;
  onAutoTableCalibration: () => void;
  onOpenSettings: () => void;
  onToggleTheme: () => void;
  onToggleLanguage: () => void;
};

export function NavigationBar({
  isDarkMode,
  labels,
  onOpenVideo,
  onOpenCamera,
  onManualTableCalibration,
  onAutoTableCalibration,
  onOpenSettings,
  onToggleTheme,
  onToggleLanguage,
}: NavigationBarProps) {
  return (
    <Layout.Header className="toolbar">
      <Space size={8}>
        <Dropdown
          menu={{
            items: [
              { key: "video", icon: <FileVideo size={16} />, label: labels.video, onClick: onOpenVideo },
              { key: "camera", icon: <Camera size={16} />, label: labels.camera, onClick: onOpenCamera },
            ],
          }}
          trigger={["click"]}
        >
          <Button aria-label={labels.openSource} icon={<FolderOpen size={17} />} />
        </Dropdown>
        <Button aria-label={labels.settings} icon={<Settings size={17} />} onClick={onOpenSettings} />
        <Dropdown
          menu={{
            items: [
              { key: "manual", icon: <Target size={16} />, label: labels.tableManualCalibration, onClick: onManualTableCalibration },
              { key: "auto", icon: <ScanSearch size={16} />, label: labels.tableAutoCalibration, onClick: onAutoTableCalibration },
            ],
          }}
          trigger={["click"]}
        >
          <Button icon={<Target size={17} />}>{labels.calibrateTable}</Button>
        </Dropdown>
      </Space>
      <Space size={8}>
        <Button
          aria-label={isDarkMode ? labels.switchToLight : labels.switchToDark}
          icon={isDarkMode ? <Sun size={17} /> : <Moon size={17} />}
          onClick={onToggleTheme}
        >
          {isDarkMode ? labels.light : labels.dark}
        </Button>
        <Button aria-label={labels.switchLanguage} icon={<Languages size={17} />} onClick={onToggleLanguage}>
          {labels.language}
        </Button>
      </Space>
    </Layout.Header>
  );
}
