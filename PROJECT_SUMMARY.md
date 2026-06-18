# 食物熱量估算系統 — 專案說明

## 系統概述

上傳食物照片 → EfficientNetB0 辨識 101 種食物類別 → USDA FoodData Central API 查詢即時營養資料 → 顯示熱量、蛋白質、碳水化合物、脂肪。

---

## 架構圖

```
瀏覽器 (food_calorie_estimator.html)
    │
    ├─ 上傳圖片
    │
    ▼
Flask 伺服器 (server.py，port 8080)
    │
    ├─ /predict  ←── POST base64 圖片
    │       │
    │       └── tf.lite.Interpreter (food101.tflite)
    │               EfficientNetB0，101 類別
    │               輸出：top-5 食物名稱 + 機率
    │
    └─ /         靜態檔案服務 (HTML/CSS/JS)

瀏覽器收到預測結果後
    │
    └── USDA FoodData Central API
            endpoint: https://api.nal.usda.gov/fdc/v1/foods/search
            查詢營養成分（熱量、蛋白質、碳水、脂肪）
```

---

## 專案檔案結構

```
oooo/
├── food_calorie_estimator.html   # 主 Web App（UI + API 整合）
├── server.py                     # Flask 後端（TFLite 推論 + 靜態服務）
├── food101.tflite                # 訓練好的模型（5.1 MB，float16 量化）
│
├── train_food101.py              # 本地訓練腳本（EfficientNetB0 Transfer Learning）
├── Food101_Colab.ipynb           # Google Colab 訓練筆記本（GPU T4）
│
├── tfjs_model/                   # TF.js Layers Model（備用，未使用）
│   ├── model.json
│   └── group1-shard[1-5]of5.bin
│
└── fix_model_json.py             # TF.js model.json 修補腳本（備用）
```

---

## 核心技術

### 模型
| 項目 | 內容 |
|------|------|
| 基礎模型 | EfficientNetB0（ImageNet 預訓練） |
| 資料集 | Food-101（101 類，共 101,000 張圖）|
| 訓練策略 | Transfer Learning 兩階段 |
| 輸入尺寸 | 224 × 224 × 3 |
| 輸出 | 101 類 Softmax 機率 |
| 最終格式 | TFLite float16 量化（5.1 MB） |
| Val Accuracy | Phase 1：72.6%（Top-1），93.2%（Top-5） |

### 訓練流程
```
Phase 1：凍結 EfficientNetB0 base，只訓練 head
  └─ GlobalAveragePooling2D → BN → Dropout(0.5)
     → Dense(512, relu) → Dropout(0.3) → Dense(101, softmax)

Phase 2：解凍最後 20 層微調（在 Google Colab T4 GPU 上跑）
  └─ 學習率降低 10 倍，繼續訓練
```

### 後端（server.py）
| 套件 | 用途 |
|------|------|
| Flask | HTTP 伺服器、路由 |
| tensorflow | tf.lite.Interpreter 跑 TFLite 模型 |
| Pillow | 圖片解碼、縮放到 224×224 |
| numpy | 陣列處理、argmax top-5 |

### 前端（food_calorie_estimator.html）
| 功能 | 實作 |
|------|------|
| 圖片上傳 / 拖曳 | HTML5 FileReader API |
| 圖片轉 base64 | Canvas.toDataURL() |
| 呼叫推論 API | fetch POST → /predict |
| 營養查詢 | fetch → USDA FoodData Central API |
| 靜態熱量備援 | 內建 101 種食物資料庫（無 API 時使用）|

---

## 啟動方式

```bash
# 安裝相依套件（第一次）
pip install flask pillow tensorflow

# 啟動伺服器
cd C:\Users\user\Downloads\oooo
python server.py

# 瀏覽器開啟
http://localhost:8080
```

---

## API 說明

### POST /predict
接受 base64 圖片，回傳 top-5 預測結果。

**Request:**
```json
{ "image": "data:image/jpeg;base64,/9j/4AAQ..." }
```

**Response:**
```json
{
  "predictions": [
    { "className": "pizza",     "probability": 0.923 },
    { "className": "bruschetta","probability": 0.041 },
    ...
  ]
}
```

### USDA FoodData Central API
```
GET https://api.nal.usda.gov/fdc/v1/foods/search
  ?query={食物名稱}
  &api_key={YOUR_KEY}
  &pageSize=5
```
回傳 nutrient ID：1008=熱量(kcal)、1003=蛋白質(g)、1005=碳水(g)、1004=脂肪(g)

---

## 101 種食物類別

apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare, beet_salad, beignets, bibimbap, bread_pudding, breakfast_burrito, bruschetta, caesar_salad, cannoli, caprese_salad, carrot_cake, ceviche, cheese_plate, cheesecake, chicken_curry, chicken_quesadilla, chicken_wings, chocolate_cake, chocolate_mousse, churros, clam_chowder, club_sandwich, crab_cakes, creme_brulee, croque_madame, cup_cakes, deviled_eggs, donuts, dumplings, edamame, eggs_benedict, escargots, falafel, filet_mignon, fish_and_chips, foie_gras, french_fries, french_onion_soup, french_toast, fried_calamari, fried_rice, frozen_yogurt, garlic_bread, gnocchi, greek_salad, grilled_cheese_sandwich, grilled_salmon, guacamole, gyoza, hamburger, hot_and_sour_soup, hot_dog, huevos_rancheros, hummus, ice_cream, lasagna, lobster_bisque, lobster_roll_sandwich, macaroni_and_cheese, macarons, miso_soup, mussels, nachos, omelette, onion_rings, oysters, pad_thai, paella, pancakes, panna_cotta, peking_duck, pho, pizza, pork_chop, poutine, prime_rib, pulled_pork_sandwich, ramen, ravioli, red_velvet_cake, risotto, samosa, sashimi, scallops, seaweed_salad, shrimp_and_grits, spaghetti_bolognese, spaghetti_carbonara, spring_rolls, steak, strawberry_shortcake, sushi, tacos, takoyaki, tiramisu, tuna_tartare, waffles
