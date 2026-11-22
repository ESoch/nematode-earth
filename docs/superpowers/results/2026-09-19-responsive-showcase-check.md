# Responsive exhibit checks, 19 September 2026

These measurements used desktop Chrome and the local Vite viewer with emulated viewport sizes and device pixel ratios (DPR). They document the September layout on that browser. Physical phones, mobile browser bars, notched displays, Safari and Firefox were outside the run.

## Opening composition

The canvas occupied the full window. A layout rectangle reserved space around the globe for the header, explanatory text and playback controls. A view offset in the perspective camera framed the globe inside that rectangle. This kept the globe circular and left the actual camera altitude available for zoom. Short landscape screens moved the explanatory text beside the globe. Very short portrait screens omitted the long quotation, which remained available on the methods page. [Framing calculation](../../../app/src/core/viewFraming.ts) · [Projection tests](../../../app/test/viewFraming.test.ts)

## Viewport results

In each row, the drawing buffer matched the CSS dimensions multiplied by DPR, capped at 2. Playback controls stayed inside the viewport, the scene rectangle cleared the surrounding controls and text, and the page had no horizontal overflow. The browser captured no unhandled promise rejections during these checks.

| CSS viewport | Emulated DPR | Drawing buffer |
| --- | ---: | --- |
| 320 × 568 | 2 | 640 × 1136 |
| 390 × 844 | 3 | 780 × 1688 |
| 768 × 1024 | 2 | 1536 × 2048 |
| 844 × 390 | 2 | 1688 × 780 |
| 568 × 320 | 2 | 1136 × 640 |
| 1024 × 768 | 1 | 1024 × 768 |
| 1440 × 900 | 2 | 2880 × 1800 |
| 2560 × 1080 | 1 | 2560 × 1080 |
| 3440 × 1440 | 1 | 3440 × 1440 |

The visual checks covered desktop, portrait, landscape and ultrawide layouts, the fully swept veil and a 300 km descent in that build. At 300 km the scene filled the window. A click on the full-screen button closed the dialog, entered native full screen and resized the canvas. Exiting full screen was also checked.

## Static page and asset checks

The methods page had working local anchors and fit the tested phone-width viewport. The model label and data dialog linked to the credits. The quotation linked to Cobb’s original passage. The built methods page retained the three.js MIT notice.

One retained, compressed 2048-face elevation file decoded byte-for-byte to its source. This spot check covered one existing asset. The production build verifies its complete binary inventory separately. [Release checklist](../../release-checklist.md)
