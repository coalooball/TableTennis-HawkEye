# TableTennis-HawkEye

A desktop table-tennis analysis app built with PyTauri, FastAPI, React, Ant Design, OpenCV, and Ultralytics YOLO. It detects human pose actions, tracks ball trajectory with a local YOLO ball model, estimates ball speed, and can mark landing points after manual table calibration.

App icon source: [Image #1](https://cn.bing.com/images/search?view=detailV2&ccid=FamT9M8t&id=490AA3646771BB6A784478973F48F58B67EB6E7F&thid=OIP.FamT9M8ttOKEMSqVBkDrJgHaHa&mediaurl=https%3a%2f%2fimg.lovepik.com%2fpng%2f20231019%2fRed-ping-pong-ball-with-racket-clipart-sports-goods-sporting_263114_wh1200.png&exph=1200&expw=1200&q=pingpang+png&mode=overlay&FORM=IQFRBA&ck=5470ECF24D39DE7BFACF96E33B7E2405&selectedIndex=0&idpp=serp&ajaxhist=0&ajaxserp=0).

## Start

```bash
uv sync
cd src/frontend
npm install
npm run build
cd ..
uv run TableTennis-HawkEye
```

Place a YOLO ball detection weight at:

```text
models/yolo11s-ball.pt
```

The default downloaded weight is Ultralytics YOLO11s COCO detection and the app filters the `sports ball` class for ball tracking. For best table-tennis accuracy, replace it with a table-tennis-ball fine-tuned YOLO weight at the same path.

Use **Calibrate Table** before analysis to reset table calibration state. Then play a video to overlay the ball trajectory and landing points.

Use **Analyze Video** to export an annotated video and trajectory CSV into `outputs/`. Speed is shown in `px/s` before table calibration and in `m/s` plus `km/h` after calibration.
