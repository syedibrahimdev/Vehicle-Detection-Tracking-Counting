Test split, 99 images; operating point conf=0.4, NMS IoU=0.5, match IoU=0.5. Brackets = 95% bootstrap CI over images.

| Class | GT boxes | Pretrained COCO recall | Fine-tuned recall | Fine-tuned precision | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| Bus | 72 | 0.44 [0.34-0.55] | 0.46 [0.35-0.57] | 0.89 [0.77-1.00] | 0.61 | 0.37 |
| Car | 552 | 0.78 [0.73-0.82] | 0.91 [0.88-0.94] | 0.80 [0.77-0.83] | 0.91 | 0.64 |
| Motor Cycle | 459 | 0.17 [0.13-0.21] | 0.72 [0.68-0.77] | 0.81 [0.76-0.85] | 0.76 | 0.33 |
| Person | 727 | 0.53 [0.49-0.57] | 0.84 [0.80-0.88] | 0.78 [0.74-0.82] | 0.85 | 0.39 |
| Rickshaw | 363 | N/A (no COCO class) | 0.89 [0.85-0.93] | 0.87 [0.84-0.91] | 0.94 | 0.62 |
| Truck | 47 | 0.23 [0.11-0.37] | 0.21 [0.11-0.33] | 0.67 [0.42-0.91] | 0.34 | 0.24 |

Overall test mAP50 = 0.734, mAP50-95 = 0.432 (model: s960_default_929531, imgsz 960, 14.2 ms/img on 0).
