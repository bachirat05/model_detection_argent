import cv2
import time
import threading
import pyttsx3
from pathlib import Path
from ultralytics import YOLO

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
model_path = str(BASE_DIR / "runs" / "detect" / "money_detector" / "weights" / "best.pt")

# ── Variables globales ─────────────────────────────────────────────────────────
total_sum       = 0.0
seen_ids        = set()
last_added_time = time.time()
said_total      = False

# ── Audio ──────────────────────────────────────────────────────────────────────
def say_audio(text):
    """Audio dans thread séparé — un moteur par thread (fix crash Windows)"""
    def speak():
        e = pyttsx3.init()
        e.setProperty('rate', 150)
        e.setProperty('volume', 1.0)
        e.say(text)
        e.runAndWait()
    threading.Thread(target=speak, daemon=True).start()

# ── Parser la valeur depuis le nom de classe ───────────────────────────────────
def parse_value(class_name):
    """'50' → 50.0  |  '0.5' → 0.5  |  erreur → 0.0"""
    try:
        return float(class_name)
    except:
        return 0.0

# ── Texte audio selon la valeur ────────────────────────────────────────────────
def valeur_en_texte(value):
    mapping = {
        0.2: "vingt centimes",
        0.5: "cinquante centimes",
        1.0: "un dirham",
        2.0: "deux dirhams",
        5.0: "cinq dirhams",
        10.0: "dix dirhams",
        20.0: "vingt dirhams",
        50.0: "cinquante dirhams",
        100.0: "cent dirhams",
        200.0: "deux cents dirhams",
    }
    return mapping.get(value, f"{value} dirhams")

# ── Bouton RETRY (clic souris) ─────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global total_sum, seen_ids, last_added_time, said_total
    if event == cv2.EVENT_LBUTTONDOWN:
        # Zone du bouton RETRY : x=10-130, y=10-60
        if 10 <= x <= 130 and 10 <= y <= 60:
            total_sum       = 0.0
            seen_ids.clear()
            said_total      = False
            last_added_time = time.time()
            say_audio("Compteur réinitialisé à zéro")
            print("🔄 Réinitialisé")

# ── Programme principal ────────────────────────────────────────────────────────
def main():
    global total_sum, seen_ids, last_added_time, said_total

    # Charger le modèle entraîné
    try:
        model = YOLO(model_path)
        print("✅ Modèle chargé :", model_path)
    except Exception as e:
        print(f"❌ Modèle introuvable : {e}")
        print("Lance d'abord : python src/train.py")
        return

    # Ouvrir la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Caméra inaccessible")
        return

    cv2.namedWindow("Money Detection")
    cv2.setMouseCallback("Money Detection", mouse_callback)

    say_audio("Système prêt. Présentez vos pièces ou billets.")
    print("📷 Caméra lancée")
    print("   → Clic sur RETRY pour remettre à zéro")
    print("   → Q ou ESC pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Détection + Tracking YOLOv8 ────────────────────────────────────────
        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            clss      = results[0].boxes.cls.cpu().numpy()
            confs     = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):

                # Ignorer les détections peu sûres
                if conf < 0.6:
                    continue

                class_name = model.names[int(cls)]
                value      = parse_value(class_name)

                # Nouvelle pièce/billet jamais compté
                if track_id not in seen_ids and value > 0:
                    seen_ids.add(track_id)
                    total_sum      += value
                    last_added_time = time.time()
                    said_total      = False

                    texte = valeur_en_texte(value)
                    say_audio(texte)
                    print(f"💰 Détecté : {class_name} = {value} DH  |  Total : {total_sum} DH")

        # ── Annonce total après 60s sans nouvelle détection ────────────────────
        if time.time() - last_added_time > 60 and not said_total and total_sum > 0:
            msg = f"Total final : {total_sum} dirhams"
            say_audio(msg)
            print(f"🔊 {msg}")
            said_total = True

        # ── Interface visuelle ─────────────────────────────────────────────────
        annotated_frame = results[0].plot()

        # Bouton RETRY (rouge)
        cv2.rectangle(annotated_frame, (10, 10), (130, 60), (0, 0, 220), -1)
        cv2.rectangle(annotated_frame, (10, 10), (130, 60), (0, 0, 150), 2)
        cv2.putText(annotated_frame, "RETRY", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Affichage total (vert)
        cv2.rectangle(annotated_frame, (10, 70), (420, 125), (0, 160, 0), -1)
        cv2.rectangle(annotated_frame, (10, 70), (420, 125), (0, 120, 0), 2)
        cv2.putText(annotated_frame, f"Total : {total_sum:.1f} DH", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        # Timer avant annonce automatique
        if not said_total and total_sum > 0:
            time_left = max(0, 60 - int(time.time() - last_added_time))
            cv2.putText(annotated_frame, f"Annonce dans : {time_left}s", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Money Detection", annotated_frame)

        # Quitter avec Q ou ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    if total_sum > 0:
        say_audio(f"Session terminée. Total : {total_sum} dirhams")
        print(f"✅ Session terminée. Total final : {total_sum} DH")

if __name__ == "__main__":
    main()
