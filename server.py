"""
server.py — 食物熱量估算系統後端
=====================================
這是整個系統的「大腦」，負責：
  1. 載入 AI 模型（EfficientNetB0，食物辨識）
  2. 接收前端傳來的食物照片，執行推論，回傳辨識結果
  3. 管理「自訂食物記憶」（讓系統記住你教它的食物）
  4. 當靜態檔案伺服器，提供 HTML 頁面給瀏覽器

執行方式：
  python server.py
  然後開啟瀏覽器 → http://localhost:8080
"""

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf   # Google 的深度學習框架，用來跑 AI 模型
import numpy as np        # 數值運算套件，處理矩陣/陣列
from PIL import Image     # Python 圖片處理套件，讀取並縮放圖片
import io, base64, os, json, time

# ── Flask 應用程式初始化 ──
# static_folder='.' 表示把當前資料夾當成靜態檔案根目錄
# 這樣 HTML、圖片等都能直接被瀏覽器取得
app = Flask(__name__, static_folder='.')

# ══════════════════════════════════════════
#  Food-101 資料集的 101 種食物類別名稱
# ══════════════════════════════════════════
# 模型訓練時用 sorted() 排序這些名稱，推論時也必須用相同順序
# 例如輸出向量第 0 個數字對應 apple_pie，第 100 個對應 waffles
CLASS_NAMES = [
    'apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare',
    'beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito',
    'bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake',
    'ceviche','cheese_plate','cheesecake','chicken_curry','chicken_quesadilla',
    'chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder',
    'club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes',
    'deviled_eggs','donuts','dumplings','edamame','eggs_benedict',
    'escargots','falafel','filet_mignon','fish_and_chips','foie_gras',
    'french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice',
    'frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich',
    'grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup',
    'hot_dog','huevos_rancheros','hummus','ice_cream','lasagna',
    'lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup',
    'mussels','nachos','omelette','onion_rings','oysters',
    'pad_thai','paella','pancakes','panna_cotta','peking_duck',
    'pho','pizza','pork_chop','poutine','prime_rib',
    'pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto',
    'samosa','sashimi','scallops','seaweed_salad','shrimp_and_grits',
    'spaghetti_bolognese','spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake',
    'sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles'
]

# ══════════════════════════════════════════
#  模型檔案路徑設定
# ══════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH  = os.path.join(BASE_DIR, 'best_final.keras')   # 完整 Keras 模型（需 GPU）
TFLITE_PATH = os.path.join(BASE_DIR, 'food101.tflite')     # 輕量化 TFLite 模型（CPU 可跑）
MEMORY_PATH = os.path.join(BASE_DIR, 'food_memory.json')   # 自訂食物記憶儲存檔

# ══════════════════════════════════════════
#  AI 模型載入
# ══════════════════════════════════════════
#
# 這個系統使用了兩種模型格式：
#
# 【Keras 模型 - best_final.keras】
#   - 完整的 EfficientNetB0 神經網路，約 20MB
#   - EfficientNetB0 是 Google 在 2019 年提出的影像辨識模型
#     特點：比傳統 ResNet 更輕量，但準確率更高
#   - 需要 NVIDIA GPU 才能發揮最快速度
#   - 在 Google Colab T4 GPU 上訓練完成
#
# 【TFLite 模型 - food101.tflite】
#   - 將 Keras 模型壓縮（量化）成更小的格式，只有 5.1MB
#   - 量化：把 32-bit 浮點數 → 16-bit，檔案縮小一半，速度更快
#   - 不需要 GPU，普通 CPU 也能執行
#   - 適合部署在手機、樹莓派等邊緣設備
#
# 程式會自動偵測：有 GPU + Keras 檔案 → 用 Keras；否則用 TFLite
gpus = tf.config.list_physical_devices('GPU')
USE_KERAS = os.path.exists(KERAS_PATH)

if USE_KERAS:
    if gpus:
        # 設定 GPU 記憶體動態成長（避免一次佔滿顯示卡記憶體）
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'✅ GPU: {[g.name for g in gpus]}')
    # 載入完整 Keras 模型
    keras_model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    # 先跑一次空白輸入（預熱），讓 GPU 完成初始化，避免第一次推論很慢
    keras_model(tf.zeros([1, 224, 224, 3]), training=False)
    print('✅ Keras 模型載入完成')
else:
    # 使用 TFLite 輕量化推論引擎
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()   # 配置記憶體空間
    # 取得輸入/輸出的規格（尺寸、資料型別等）
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('✅ TFLite 模型載入完成')


# ══════════════════════════════════════════
#  自訂食物記憶（Custom Food Memory）
# ══════════════════════════════════════════
#
# 這個功能解決了「模型不認識台灣食物」的問題。
# 例如模型看到雞塊可能猜成 chicken_wings，但你可以告訴它「這是雞塊」
# 系統會記住這張圖的「指紋」，下次看到類似圖片就能正確辨識。
#
# 技術原理：
#   用 AI 模型的「最終輸出向量」當作圖片指紋（Embedding）
#   每張圖會產生 101 個數字（每種食物的機率）
#   用「餘弦相似度」比較兩張圖的指紋，相似度 > 0.92 就判定為同一種食物

def load_memory():
    """從 JSON 檔案讀取所有記憶的食物"""
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []  # 還沒有記憶時回傳空串列

def save_memory(memory):
    """把食物記憶儲存回 JSON 檔案（保留中文，不轉 ASCII）"""
    with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def cosine_similarity(a, b):
    """
    計算兩個向量的餘弦相似度（Cosine Similarity）

    公式：cos(θ) = (A · B) / (|A| × |B|)

    結果範圍 0 ~ 1：
      1.0 = 完全相同
      0.9 = 非常相似
      0.5 = 有點像
      0.0 = 完全不同

    為什麼用餘弦相似度而不是歐式距離？
    因為我們只在乎「方向」（食物種類的分佈），不在乎「大小」
    """
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)   # 兩向量長度的乘積
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def get_embedding(arr):
    """
    用 AI 模型取得圖片的特徵向量（Embedding / 指紋）

    輸入：arr — 形狀 [1, 224, 224, 3] 的圖片陣列
    輸出：長度 101 的浮點數串列，代表每種食物的機率

    這 101 個數字就是這張圖的「指紋」
    不同食物的指紋差異很大，同樣食物的指紋非常接近
    """
    if USE_KERAS:
        # Keras 模型：直接呼叫，training=False 關閉 Dropout 層
        probs = keras_model(arr, training=False).numpy()[0]
    else:
        # TFLite 推論：設定輸入 → 執行 → 讀取輸出
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
    return probs.tolist()

def check_memory(embedding, threshold=0.92):
    """
    把當前圖片的指紋與記憶庫中所有食物比對

    參數：
      embedding — 當前圖片的 101 維指紋向量
      threshold — 相似度門檻，預設 0.92（越高越嚴格）

    回傳：
      如果找到相似食物 → {'label': '雞塊', 'nutrition_query': 'chicken nuggets', 'similarity': 0.95}
      找不到          → None
    """
    memory = load_memory()
    if not memory:
        return None  # 記憶庫空的，直接回傳

    best_sim, best_item = 0.0, None
    for item in memory:
        sim = cosine_similarity(embedding, item['embedding'])
        if sim > best_sim:
            best_sim, best_item = sim, item

    # 只有超過門檻才採用，避免誤判
    if best_sim >= threshold:
        return {
            'label': best_item['label'],
            'nutrition_query': best_item.get('nutrition_query', ''),
            'similarity': best_sim
        }
    return None


# ══════════════════════════════════════════
#  圖片處理工具
# ══════════════════════════════════════════

def decode_image(b64_str):
    """
    把瀏覽器傳來的 Base64 圖片字串轉成模型能讀的陣列

    Base64 是一種把圖片二進位資料轉成文字的編碼方式
    格式通常是："data:image/jpeg;base64,/9j/4AAQ..."

    處理步驟：
      1. 去掉開頭的 "data:image/jpeg;base64," 標頭
      2. Base64 解碼 → 取得原始圖片位元組
      3. PIL 開啟圖片，轉成 RGB（去掉透明通道）
      4. 縮放到 224×224（模型要求的輸入尺寸）
      5. 轉成 NumPy 陣列，形狀 [1, 224, 224, 3]
         （1 = batch size, 224×224 = 圖片尺寸, 3 = RGB 三個頻道）
    """
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]   # 去掉 data:image/... 標頭
    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    arr = np.array(pil_img.resize((224, 224)), dtype=np.float32)[np.newaxis]
    return pil_img, arr

def run_inference(arr):
    """
    執行 AI 推論，回傳 Top-5 預測結果

    輸入：arr — 形狀 [1, 224, 224, 3] 的圖片陣列
    輸出：
      predictions — [{'className': 'pizza', 'probability': 0.923}, ...]（前5名）
      embedding   — 長度 101 的指紋向量（給記憶比對用）

    模型輸出是 101 個機率值（Softmax），全部加起來 = 1
    我們取最大的前 5 個，就是 Top-5 預測
    """
    embedding = get_embedding(arr)
    probs = np.array(embedding)
    # argsort 回傳「從小到大的索引」，[::-1] 反轉成「從大到小」，[:5] 取前 5
    top5_idx = np.argsort(probs)[::-1][:5]
    predictions = [{'className': CLASS_NAMES[i], 'probability': float(probs[i])}
                   for i in top5_idx]
    return predictions, embedding


# ══════════════════════════════════════════
#  API 端點（必須定義在 catch-all 靜態路由之前）
# ══════════════════════════════════════════
#
# RESTful API 設計：
#   GET  /ping         → 確認伺服器存活
#   POST /predict      → 上傳圖片，取得辨識結果
#   POST /remember     → 儲存自訂食物記憶
#   GET  /memory       → 查看所有已記憶的食物
#   DELETE /memory/<>  → 刪除某個記憶

@app.route('/ping')
def ping():
    """
    健康檢查端點：前端載入時會先呼叫這個
    確認後端伺服器已啟動，模型已載入完成
    """
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    食物辨識主要端點

    請求格式（JSON）：
      { "image": "data:image/jpeg;base64,..." }

    回應格式（JSON）：
      一般情況：
        { "predictions": [{"className": "pizza", "probability": 0.92}, ...] }

      有記憶符合時：
        {
          "predictions": [...],        ← 模型原始預測（Top-5）
          "memory_match": {            ← 記憶庫找到的符合項目
            "label": "雞塊",
            "nutrition_query": "chicken nuggets",
            "similarity": 0.95
          }
        }

    處理流程：
      圖片 → decode → run_inference → check_memory → 回傳結果
    """
    try:
        data = request.get_json(force=True)
        _, arr = decode_image(data.get('image', ''))
        predictions, embedding = run_inference(arr)

        # 先查記憶庫，如果有符合就附上記憶結果
        memory_match = check_memory(embedding)
        if memory_match:
            return jsonify({
                'predictions': predictions,
                'memory_match': memory_match   # {label, nutrition_query, similarity}
            })

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/remember', methods=['POST'])
def remember():
    """
    儲存自訂食物記憶

    請求格式（JSON）：
      {
        "image": "data:image/jpeg;base64,...",   ← 食物圖片
        "label": "雞塊",                          ← 你想叫它什麼名字（中文OK）
        "nutrition_query": "chicken nuggets"     ← 熱量查詢用的英文詞（選填）
      }

    處理邏輯：
      1. 解碼圖片，取得 101 維指紋
      2. 如果記憶庫中已有相似度 >= 0.96 的項目 → 更新（覆蓋舊名稱）
      3. 否則 → 新增一筆記憶
      4. 儲存到 food_memory.json

    為什麼存指紋而不是存圖片？
      指紋只有 101 個數字（約 800 bytes），圖片可能幾百 KB
      而且指紋比較特徵更精準，不受光線/角度小幅變化影響
    """
    try:
        data = request.get_json(force=True)
        label = data.get('label', '').strip()
        if not label:
            return jsonify({'error': '請輸入食物名稱'}), 400

        _, arr = decode_image(data.get('image', ''))
        embedding = get_embedding(arr)

        memory = load_memory()

        # 檢查是否已存在非常相似的記憶（相似度 0.96 以上 = 幾乎同一張圖）
        updated = False
        for item in memory:
            if cosine_similarity(embedding, item['embedding']) >= 0.96:
                item['label'] = label
                item['nutrition_query'] = data.get('nutrition_query', '').strip()
                item['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                updated = True
                break

        if not updated:
            memory.append({
                'label': label,
                'nutrition_query': data.get('nutrition_query', '').strip(),
                'embedding': embedding,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })

        save_memory(memory)
        return jsonify({'status': 'ok', 'total': len(memory), 'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/memory', methods=['GET'])
def list_memory():
    """列出記憶庫中所有自訂食物（不回傳指紋向量，只要名稱和時間）"""
    memory = load_memory()
    return jsonify({
        'total': len(memory),
        'items': [{'label': m['label'], 'created_at': m.get('created_at', '')} for m in memory]
    })


@app.route('/memory/<label>', methods=['DELETE'])
def delete_memory(label):
    """刪除指定名稱的食物記憶"""
    memory = load_memory()
    before = len(memory)
    memory = [m for m in memory if m['label'] != label]
    save_memory(memory)
    return jsonify({'deleted': before - len(memory)})


# ══════════════════════════════════════════
#  靜態檔案服務（必須放在 API 路由之後）
# ══════════════════════════════════════════
#
# 為什麼 Flask 要同時當靜態檔案伺服器？
# 這樣只需要執行一個 python 指令，
# Flask 就同時負責：
#   - 提供 HTML 頁面給瀏覽器（靜態服務）
#   - 處理 /predict 等 API 請求（動態服務）
# 不需要另外安裝 Nginx 或 Apache

@app.route('/')
def index():
    """首頁：直接回傳 HTML 主頁面"""
    return send_from_directory('.', 'food_calorie_estimator.html')

# API 路徑集合：防止 API 路徑被 catch-all 路由攔截導致錯誤
API_PATHS = {'ping', 'predict', 'remember', 'memory'}

@app.route('/<path:filename>', methods=['GET', 'HEAD'])
def static_files(filename):
    """
    通用靜態檔案路由（Catch-all）

    處理 HTML 裡用到的所有靜態資源請求
    例如：/training_metrics.png、/PROJECT_SUMMARY.md 等

    特殊防護：如果路徑是 API 名稱（predict、remember 等），
    回傳 404 而不是嘗試讀取檔案，避免路由衝突
    """
    top = filename.split('/')[0]
    if top in API_PATHS:
        return jsonify({'error': 'not found'}), 404
    return send_from_directory('.', filename)


# ══════════════════════════════════════════
#  程式進入點
# ══════════════════════════════════════════
if __name__ == '__main__':
    print('🌐 伺服器啟動: http://localhost:8080')
    print('📌 按 Ctrl+C 停止伺服器')
    # debug=False：正式模式（不自動重載、不顯示錯誤詳情給使用者）
    # threaded=True：允許同時處理多個請求（多執行緒）
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
